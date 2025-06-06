from egt309_insight_boys_kedro.pipelines.pipeline import create_pipeline


def register_pipelines() -> dict:
    return {
        "__default__": create_pipeline(),
    }
