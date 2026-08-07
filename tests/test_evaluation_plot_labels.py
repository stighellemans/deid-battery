import matplotlib.pyplot as plt
import pandas as pd

from deid_battery._vendor import evaluation_plots as ep


def test_non_pii_component_legend_uses_descriptive_labels() -> None:
    summary = pd.DataFrame(
        {
            "source": ["Example"],
            "core_pii_recall": [0.95],
            "machine_only_redaction_rate": [0.01],
            "boundary_overflow_redaction_rate": [0.002],
            "non_pii_redaction_rate": [0.012],
        }
    )

    figure = ep.plot_recall_and_non_pii_redaction(summary)
    try:
        labels = figure.axes[1].get_legend_handles_labels()[1]
        assert labels == [
            "False-positive spans",
            "PII boundary extensions",
        ]
    finally:
        plt.close(figure)
