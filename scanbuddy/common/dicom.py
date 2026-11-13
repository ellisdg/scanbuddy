import logging
from typing import Any, Optional, Tuple, Union

from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement
from pydicom.sequence import Sequence
from pydicom.tag import Tag as DicomTag, BaseTag

logger = logging.getLogger(__name__)

TagLike = Union[Tuple[int, int], BaseTag, int]
PathStep = Union[int, TagLike]


def _to_tag(tag: TagLike) -> DicomTag:
    """Normalize various tag representations to a pydicom Tag (BaseTag)."""
    # Already a Tag/BaseTag or an int
    if isinstance(tag, (BaseTag, int)):
        return DicomTag(tag)
    # Tuple of (group, element)
    if isinstance(tag, tuple) and len(tag) == 2:
        return DicomTag(tag)
    # Fallback: let pydicom try to interpret (may raise)
    return DicomTag(tag)


def get_tag(ds: Dataset, tag: TagLike, default: Any = None, *, required: bool = False) -> Any:
    """
    Safely get a tag's value from a Dataset.
    - Returns default if missing (unless required=True, then raises KeyError).
    - Accepts tuple (0x0018, 0x1250), BaseTag, or int.
    """
    elem = ds.get(_to_tag(tag))
    if elem is None:
        if required:
            # Use normalized DicomTag for consistent formatting
            t = _to_tag(tag)
            raise KeyError(f"Missing DICOM tag {t}")
        return default
    return elem.value


def get_path(ds: Dataset, *path: PathStep, default: Any = None) -> Any:
    """
    Safely traverse a nested path of DICOM elements.
    Path consists of tags and/or integer indices. Examples:
      get_path(ds, (0x0018, 0x1250))
      get_path(ds, (0x5200, 0x9229), 0, (0x0018, 0x9042), 0, (0x0018, 0x1250))
    Returns default if any step is missing or invalid.
    """
    cur: Any = ds
    try:
        for step in path:
            # If cur is DataElement, unwrap to its value first
            if isinstance(cur, DataElement):
                cur = cur.value

            if isinstance(step, int):
                # Expect a Sequence or list-like
                if isinstance(cur, Sequence) or isinstance(cur, list):
                    cur = cur[step]
                else:
                    return default
            else:
                # A tag step (tuple/BaseTag/int)
                t = _to_tag(step)
                if isinstance(cur, Dataset):
                    elem = cur.get(t)
                elif isinstance(cur, dict):
                    elem = cur.get(t)  # defensive
                else:
                    return default
                if elem is None:
                    return default
                cur = elem.value
        if isinstance(cur, DataElement):
            return cur.value
        return cur
    except Exception as e:
        step_str = locals().get('step', '?')
        logger.debug(f"get_path failed at step {step_str}: {e}")
        return default


def get_receive_coil_name(ds: Dataset) -> str:
    """
    Return receive coil name using several strategies:
      1) Classic tag (0018,1250) Receive Coil Name
      2) Enhanced MR: Shared/Per-frame -> MRReceiveCoilSequence -> ReceiveCoilName
      3) Vendor private fallbacks (Siemens, Philips, etc.)
    """
    # 1) Classic
    v = get_tag(ds, (0x0018, 0x1250))
    if v:
        return v

    # 2) Enhanced MR functional groups
    # Shared Functional Groups
    v = get_path(ds, (0x5200, 0x9229), 0, (0x0018, 0x9042), 0, (0x0018, 0x1250))
    if v:
        return v
    # Per-frame Functional Groups (first frame)
    v = get_path(ds, (0x5200, 0x9230), 0, (0x0018, 0x9042), 0, (0x0018, 0x1250))
    if v:
        return v

    # 3) Vendor private examples (not exhaustive)
    for t in ((0x0051, 0x100f), (0x2001, 0x9057)):
        pv = get_tag(ds, t)
        if pv:
            return pv

    return "[UNKNOWN-COIL]"


def get_table_position_value(ds: Dataset) -> Optional[str]:
    """
    Return the raw table position string from a known private sequence path,
    if available. This function preserves the existing behavior where callers
    may use the last character of the string.
    """
    return get_path(ds, (0x5200, 0x9230), 0, (0x0021, 0x11fe), 0, (0x0021, 0x1145))


def get_coil_elements_value(ds: Dataset) -> Optional[str]:
    """
    Return the coil elements string from a known private sequence path, if available.
    """
    return get_path(ds, (0x5200, 0x9230), 0, (0x0021, 0x11fe), 0, (0x0021, 0x114f))


def get_num_vols(ds: Dataset, default: Optional[int] = None) -> Optional[int]:
    """
    Try to determine the number of volumes/images in the acquisition.
    Priority:
      - (0020,0105) Number of Temporal Positions
      - (0020,1002) Images in Acquisition
      - fallback to provided default
    """
    for tag in ((0x0020, 0x0105), (0x0020, 0x1002)):
        v = get_tag(ds, tag)
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                pass
    return default
