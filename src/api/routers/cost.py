"""Cost analysis router — exposes per-model USD cost breakdown."""
from fastapi import APIRouter, Depends

from ..dependencies import get_settings
from ..model_cost.analysis import get_cost_analysis

router = APIRouter(prefix="/api/cost", tags=["model cost"])


@router.get("/analysis")
def get_cost_analysis_endpoint(settings=Depends(get_settings)):
    """Return a per-model token usage and cost breakdown.

    Returns:
        A dict with ``note``, ``models`` list, and ``grand_total`` keys.
    """
    return get_cost_analysis()
