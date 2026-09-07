"""Convert the ICRA2024 QRC simulation map from URDF into mjlab terrain assets.

The upstream map (https://github.com/teamgrit-lab/ICRA2024_Quadruped_Robot_Challenges) is a URDF
with 171 fixed links, each a single mesh used for both visual and collision. MuJoCo collides meshes
by their convex hull, and 8 of the 14 unique meshes are strongly non-convex (a hull of the pallet
step with its pipe fills 10x its real volume), so the raw meshes cannot be used directly.

The meshes are face soups with unmerged vertices. After merging vertices, every mesh splits into
connected components that are individually convex, and 46 of the 52 resulting parts are exact
oriented bounding boxes. So the conversion emits box geoms wherever possible and convex-hull meshes
for the remaining 6 parts -- no approximate decomposition (CoACD/VHACD) is needed.

Output lands in ``lorl_mjlab/terrains/assets/icra/``: one MJCF per map variant plus the handful of
convex-part OBJs, consumed by ``lorl_mjlab.terrains.icra_map``.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro
from trimesh.transformations import euler_matrix, quaternion_from_matrix

from lorl_mjlab.terrains.icra_map import EXTENT_FIELD, ICRA_ASSET_DIR, SPAWN_FIELD, VARIANTS

REPO_URL = "https://github.com/teamgrit-lab/ICRA2024_Quadruped_Robot_Challenges"
MAP_SUBDIR = "ICRA2024_QRC_Simulation_Map"

# Kinds of convex part a source mesh decomposes into. Local to the importer: the emitted MJCF says
# ``type="box"`` or ``type="mesh"`` directly.
KIND_BOX = 0
KIND_MESH = 1

# Geom class every course geom is emitted under, so colour and group are stated once.
GEOM_CLASS = "icra"

# Robot start pad, in the raw URDF world frame. Taken from the lorl_isaaclab ICRA play configs,
# which spawn the robot at this corner of the course.
START_XY_URDF = (6.0, -4.3)
# The pad sits at the +x corner, so the robot faces down -x into the course.
START_YAW = math.pi

# A part is emitted as a box geom when its convex hull fills at least this much of its own oriented
# bounding box. In practice the box-like parts score 1.0 and the rest score below 0.6.
BOX_FILL_THRESHOLD = 0.98
# A part emitted as a convex mesh must fill at least this much of its own hull, i.e. the hull must
# not invent geometry the robot could stand on.
MIN_HULL_FILL = 0.95
# Connected components below this convex-hull volume are open shells with no interior; upstream has
# two of them (in pallet_step and pallet_step_with_pipe).
MIN_PART_VOLUME = 1e-7
# Vertex merge tolerance, in decimal digits. The meshes are authored in millimetre multiples, so
# this is far tighter than any real feature yet still welds the duplicated face-soup vertices.
MERGE_DIGITS = 5

DEFAULT_RGBA = (0.92, 0.93, 0.90, 1.0)

# k_rail_diagonal is a single connected component shaped like a chevron: two diagonal slabs meeting
# at an apex on the y=0 plane. Its hull fills the whole 1.1 x 2.3 m footprint, which would place an
# invisible wall across the course, so it is cut along its plane of symmetry into two convex halves.
CHEVRON_MESH = "k_rail_diagonal"


@dataclass(frozen=True)
class Args:
    repo: Path | None = None
    """Path to an existing checkout of the upstream repo. Cloned to a temp dir when omitted."""


@dataclass(frozen=True, eq=False)
class Part:
    """One convex piece of a source mesh, in that mesh's local frame."""

    name: str
    """Stable identifier, ``<source mesh>_partNN``. Doubles as the OBJ basename for mesh parts."""
    kind: int
    size: np.ndarray
    """Box half-extents, in meters. Zeros for mesh parts."""
    transform: np.ndarray
    """4x4 pose of the box within the mesh frame. Identity for mesh parts."""
    hull: trimesh.Trimesh | None
    """Convex hull geometry for mesh parts, already in the mesh frame. None for box parts."""


@dataclass(frozen=True)
class Link:
    """One placed instance of a source mesh in the URDF world frame."""

    mesh: str
    transform: np.ndarray
    rgba: tuple[float, float, float, float]


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _resolve_repo(repo: Path | None, workdir: Path) -> tuple[Path, str]:
    """Return the map package root and the upstream commit it came from."""
    if repo is None:
        clone = workdir / "upstream"
        print(f"[icra] cloning {REPO_URL}")
        _run(["git", "clone", "--depth", "1", REPO_URL, str(clone)])
        root = clone
    else:
        root = repo.expanduser().resolve()
    commit = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    map_root = root / MAP_SUBDIR
    if not (map_root / "urdf").is_dir():
        raise FileNotFoundError(f"{map_root} does not look like the ICRA map package")
    return map_root, commit


def _split_chevron(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Cut the k-rail chevron along its y=0 plane of symmetry into two convex halves."""
    halves = []
    for sign in (1.0, -1.0):
        half = mesh.slice_plane(plane_origin=(0.0, 0.0, 0.0), plane_normal=(0.0, sign, 0.0))
        if half is None or len(half.vertices) < 4:
            raise RuntimeError(f"{CHEVRON_MESH}: symmetry-plane cut produced an empty half")
        halves.append(half.convex_hull)
    return halves


def _decompose(path: Path) -> list[trimesh.Trimesh]:
    """Split one source mesh into convex pieces."""
    mesh = trimesh.load(path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    mesh.merge_vertices(digits_vertex=MERGE_DIGITS)

    pieces: list[trimesh.Trimesh] = []
    for piece in mesh.split(only_watertight=False):
        # Open shells have no volume, so trimesh divides by zero deriving their mass properties.
        with np.errstate(invalid="ignore", divide="ignore"):
            if piece.convex_hull.volume < MIN_PART_VOLUME:
                continue
        pieces.extend(_split_chevron(piece) if path.stem == CHEVRON_MESH else [piece])
    return pieces


def _to_part(piece: trimesh.Trimesh, name: str) -> Part:
    """Classify a convex piece as either a box geom or a convex mesh geom.

    Raises if the piece is not faithfully represented by either. A piece may legitimately fall short
    of its own hull -- ``crate_board`` is a platform with shallow recesses -- but only if it still
    fills its oriented bounding box, in which case a box geom reproduces the walkable surface.
    """
    hull = piece.convex_hull
    obb = hull.bounding_box_oriented
    if hull.volume / obb.volume > BOX_FILL_THRESHOLD:
        return Part(
            name=name,
            kind=KIND_BOX,
            size=np.asarray(obb.primitive.extents, dtype=np.float64) / 2.0,
            transform=np.asarray(obb.primitive.transform, dtype=np.float64),
            hull=None,
        )
    fill = abs(piece.volume) / hull.volume
    if fill < MIN_HULL_FILL:
        raise RuntimeError(f"{name}: neither a box nor convex (fills {fill:.2f} of its hull)")
    return Part(name=name, kind=KIND_MESH, size=np.zeros(3), transform=np.eye(4), hull=hull)


def _parse_urdf(path: Path) -> list[Link]:
    """Read the placed collision meshes out of a map URDF.

    Every link is the child of exactly one fixed joint off ``world``, and carries exactly one
    collision mesh at the link origin.
    """
    root = ET.parse(path).getroot()
    joints = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        origin = joint.find("origin")
        assert child is not None and origin is not None
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        transform = euler_matrix(*rpy, axes="sxyz")
        transform[:3, 3] = xyz
        joints[child.get("link")] = transform

    links = []
    for link in root.findall("link"):
        mesh = link.find("collision/geometry/mesh")
        if mesh is None:
            continue
        name = link.get("name")
        assert name is not None
        filename = mesh.get("filename")
        assert filename is not None
        color = link.find("visual/material/color")
        rgba = DEFAULT_RGBA
        if color is not None and color.get("rgba"):
            values = [float(v) for v in color.get("rgba", "").split()]
            rgba = (values[0], values[1], values[2], values[3])
        links.append(Link(mesh=Path(filename).stem, transform=joints[name], rgba=rgba))
    return links


def _corners(part: Part, transform: np.ndarray) -> np.ndarray:
    """World-frame corner points of a placed part, used to size and align the course."""
    if part.kind == KIND_BOX:
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        points = signs * part.size
        pose = transform @ part.transform
    else:
        assert part.hull is not None
        points = np.asarray(part.hull.vertices)
        pose = transform
    return points @ pose[:3, :3].T + pose[:3, 3]


# Values below this are numerical dust from the upstream rpy angles and the box fitting, not real
# offsets; a nanometre is far below any real feature.
SNAP_TOLERANCE = 1e-9
# Emitted coordinates are rounded to this many decimals, i.e. to 0.1 um and 0.1 urad. The upstream
# meshes are single precision, so this drops float32 dust -- 0.30000001 back to 0.3 -- at a cost far
# below any real feature.
DECIMALS = 7


def _fmt(values) -> str:
    """Format a vector as an MJCF attribute."""
    array = np.asarray(values, dtype=np.float64)
    array = np.where(np.abs(array) < SNAP_TOLERANCE, 0.0, array).round(DECIMALS)
    array = array + 0.0  # Collapse the -0.0 that rounding leaves behind.
    return " ".join(f"{v:.10g}" for v in array)


def _variant_xml(variant: str, links: list[Link], parts: dict[str, list[Part]], commit: str) -> str:
    """Expand placed links into MJCF: one geom per convex part, at its pose in the course frame."""
    points = np.concatenate([_corners(p, link.transform) for link in links for p in parts[link.mesh]])
    lo, hi = points.min(axis=0), points.max(axis=0)
    # Centre the course in XY and drop it onto z=0 so it sits on the terrain patch floor.
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])

    root = ET.Element("mujoco", model=f"icra_{variant}")
    root.append(ET.Comment(f" Generated from {REPO_URL}\n       at {commit}. Regenerate `uv run convert-icra-map`. "))
    ET.SubElement(root, "compiler", angle="radian", meshdir=".")

    # What the terrain module reads off this file, so it never has to compile the course to size the
    # patch or place the robot.
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name=EXTENT_FIELD, data=_fmt(hi - lo))
    spawn = [START_XY_URDF[0] + shift[0], START_XY_URDF[1] + shift[1], START_YAW]
    ET.SubElement(custom, "numeric", name=SPAWN_FIELD, data=_fmt(spawn))

    asset = ET.SubElement(root, "asset")
    for name in sorted({p.name for link in links for p in parts[link.mesh] if p.kind == KIND_MESH}):
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.obj")

    default = ET.SubElement(root, "default")
    course = ET.SubElement(default, "default", {"class": GEOM_CLASS})
    # Group 0 is what the task's foot height scanners ray-cast against.
    ET.SubElement(course, "geom", group="0", rgba=_fmt(DEFAULT_RGBA))

    world = ET.SubElement(root, "worldbody")
    for index, link in enumerate(links):
        for part in parts[link.mesh]:
            # A box carries its own pose within the source mesh; a mesh part is already in that frame.
            pose = link.transform @ part.transform if part.kind == KIND_BOX else link.transform
            # Named so the terrain test can match a placed geom back to the part it came from.
            geom = ET.SubElement(world, "geom", {"name": f"{part.name}__{index:03d}", "class": GEOM_CLASS})
            if part.kind == KIND_BOX:
                geom.set("type", "box")
                geom.set("size", _fmt(part.size))
            else:
                geom.set("type", "mesh")
                geom.set("mesh", part.name)
            geom.set("pos", _fmt(pose[:3, 3] + shift))
            quat = quaternion_from_matrix(pose)
            if not np.allclose(quat, (1.0, 0.0, 0.0, 0.0)):
                geom.set("quat", _fmt(quat))
            if link.rgba != DEFAULT_RGBA:
                geom.set("rgba", _fmt(link.rgba))

    ET.indent(root, "    ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _convert(map_root: Path, commit: str, out_dir: Path) -> None:
    mesh_dir = map_root / "meshes" / "visual"
    urdf_dir = map_root / "urdf"
    out_dir.mkdir(parents=True, exist_ok=True)

    links_by_variant = {v: _parse_urdf(urdf_dir / f"map_{v}.urdf") for v in VARIANTS}
    used = sorted({link.mesh for links in links_by_variant.values() for link in links})

    parts: dict[str, list[Part]] = {}
    mesh_files: list[str] = []
    for mesh in used:
        pieces = _decompose(mesh_dir / f"{mesh}.obj")
        parts[mesh] = [_to_part(piece, f"{mesh}_part{i:02d}") for i, piece in enumerate(pieces)]
        for part in parts[mesh]:
            if part.hull is None:
                continue
            part.hull.export(out_dir / f"{part.name}.obj", file_type="obj", include_color=False)
            mesh_files.append(part.name)

    n_box = sum(p.kind == KIND_BOX for ps in parts.values() for p in ps)
    print(f"[icra] {len(used)} source meshes -> {n_box} box parts, {len(mesh_files)} convex mesh parts")

    for variant, links in links_by_variant.items():
        xml = _variant_xml(variant, links, parts, commit)
        (out_dir / f"{variant}.xml").write_text(xml)
        n_geoms = sum(len(parts[link.mesh]) for link in links)
        print(f"[icra] {variant}: {len(links)} links -> {n_geoms} geoms")


def main() -> None:
    args = tyro.cli(Args)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        map_root, commit = _resolve_repo(args.repo, workdir)
        _convert(map_root, commit, ICRA_ASSET_DIR)
    print(f"[icra] wrote {ICRA_ASSET_DIR}")


if __name__ == "__main__":
    main()
