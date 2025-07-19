#!/usr/bin/env python3  # noqa: EXE001
"""
Production-Ready Prometheus Telemetry Example

Demonstrates integrating gemini-batch telemetry with Prometheus for production monitoring.
Uses live API calls to show real metrics collection.

---

### Adapting to Production

To use with a real Prometheus server:

1. Install: `pip install prometheus-client`

2. Replace MockPrometheusClient with real client:
   ```python
   from prometheus_client import start_http_server
   start_http_server(8000)  # Exposes /metrics endpoint
   ```

3. Keep PrometheusReporter unchanged - it works with both mock and real clients.
"""  # noqa: D212, D415, E501

import os

os.environ["GEMINI_TELEMETRY"] = "1"

from typing import Any, Dict, Tuple  # noqa: UP035

from gemini_batch import BatchProcessor
from gemini_batch.telemetry import TelemetryContext, TelemetryReporter


class MockPrometheusClient:
    """Mock Prometheus client for demonstration."""

    def __init__(self):  # noqa: ANN204, D107
        self._metrics = {}

    def _get_metric_key(self, name: str, labels: Dict[str, str]) -> str:  # noqa: UP006
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def Counter(self, name: str, description: str = "", labelnames: Tuple[str, ...] = ()):  # noqa: ANN201, ARG002, D102, E501, N802, UP006
        self._metrics.setdefault(name, {"type": "COUNTER", "desc": description, "values": {}})  # noqa: E501
        client_instance = self
        class LabeledCounter:
            def labels(self, **labels):  # noqa: ANN003, ANN202
                class FinalCounter:
                    def inc(self, amount: int = 1):  # noqa: ANN202
                        key = client_instance._get_metric_key(name, labels)  # noqa: SLF001
                        client_instance._metrics[name]["values"].setdefault(key, 0)  # noqa: SLF001
                        client_instance._metrics[name]["values"][key] += amount  # noqa: SLF001
                        print(f"[prometheus] COUNTER {key}: +{amount}")
                return FinalCounter()
        return LabeledCounter()

    def Histogram(self, name: str, description: str = "", labelnames: Tuple[str, ...] = ()):  # noqa: ANN201, ARG002, D102, E501, N802, UP006
        self._metrics.setdefault(name, {"type": "HISTOGRAM", "desc": description, "values": {}})  # noqa: E501
        client_instance = self
        class LabeledHistogram:
            def labels(self, **labels):  # noqa: ANN003, ANN202
                class FinalHistogram:
                    def observe(self, value: float):  # noqa: ANN202
                        key = client_instance._get_metric_key(name, labels)  # noqa: SLF001
                        client_instance._metrics[name]["values"].setdefault(key, {"sum": 0.0, "count": 0})  # noqa: E501, SLF001
                        client_instance._metrics[name]["values"][key]["sum"] += value  # noqa: SLF001
                        client_instance._metrics[name]["values"][key]["count"] += 1  # noqa: SLF001
                        print(f"[prometheus] HISTOGRAM {key}: observed {value:.4f}s")
                return FinalHistogram()
        return LabeledHistogram()

    def print_report(self):  # noqa: ANN201
        """Prints the final state of all collected metrics in Prometheus format."""
        print("\n--- Mock Prometheus Final State ---")
        if not self._metrics:
            print("No metrics were recorded.")
            return
        for name, data in self._metrics.items():
            print(f"# HELP {name} {data['desc']}")
            print(f"# TYPE {name} {data['type'].lower()}")
            for key, value in data["values"].items():
                if data["type"] == "COUNTER":
                    print(f"{key} {value}")
                elif data["type"] == "HISTOGRAM":
                    # Correctly format histogram output
                    metric_name_without_labels = key.split('{')[0]  # noqa: Q000
                    labels_part = key.replace(metric_name_without_labels, '')  # noqa: Q000
                    print(f"{metric_name_without_labels}_sum{labels_part} {value['sum']:.4f}")  # noqa: E501
                    print(f"{metric_name_without_labels}_count{labels_part} {value['count']}")  # noqa: E501
        print("-----------------------------------")


class PrometheusReporter(TelemetryReporter):
    """TelemetryReporter that sends metrics to Prometheus."""

    def __init__(self, client: Any, model_name: str):  # noqa: ANN204, ANN401, D107
        self.prometheus = client
        self.model_name = model_name
        self.api_call_duration = self.prometheus.Histogram(
            "gemini_api_duration_seconds",
            "Duration of underlying Gemini API calls, including retries.",
            labelnames=("model",),
        )
        self.batch_operations = self.prometheus.Counter(
            "gemini_batch_operations_total",
            "Total number of batch processing operations",
            labelnames=("status",),
        )
        self.token_usage = self.prometheus.Counter(
            "gemini_tokens_total",
            "Total tokens processed (prompt + output)",
            labelnames=("model",),
        )

    def record_timing(self, scope: str, duration: float, **metadata):  # noqa: ANN003, ANN201, ARG002
        """Maps timing data to Prometheus Histograms."""
        if scope.endswith("client.api_call_with_retry"):
            self.api_call_duration.labels(model=self.model_name).observe(duration)

    def record_metric(self, scope: str, value: Any, **metadata):  # noqa: ANN003, ANN201, ANN401, ARG002
        """Maps data to Prometheus Counters."""
        if scope.endswith("batch_success"):
            self.batch_operations.labels(status="success").inc(int(value))
        elif scope.endswith("batch_failures"):
            self.batch_operations.labels(status="failure").inc(int(value))
        elif scope.endswith("total_tokens"):
            self.token_usage.labels(model=self.model_name).inc(int(value))


def main():  # noqa: ANN201
    """Main function to run the telemetry simulation."""
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key to run this live example.")
        return

    temp_processor = BatchProcessor()
    model_name = temp_processor.client.config_manager.model
    print(f"Detected model from configuration: {model_name}")

    mock_prometheus_client = MockPrometheusClient()
    prometheus_reporter = PrometheusReporter(
        client=mock_prometheus_client, model_name=model_name  # noqa: COM812
    )
    tele_context = TelemetryContext(prometheus_reporter)

    processor = BatchProcessor(telemetry_context=tele_context)

    content = (
        "The Gemini Batch framework provides efficient analysis of multimodal content."
    )
    questions = [
        "What is the main benefit of this framework?",
        "How does it improve efficiency?",
        "What are some key features mentioned?",
    ]

    print("\nStarting batch processing with Prometheus telemetry enabled...")
    try:
        results = processor.process_questions(content, questions)
        print(f"Successfully processed {results.get('question_count')} questions.")
        for i, answer in enumerate(results.get("answers", []), 1):
            print(f"  Answer {i}: {answer[:80]}...")

    except Exception as e:  # noqa: BLE001
        print(f"\nAn error occurred during processing: {e}")
        print("Please ensure your API key is valid and has access to the configured model.")  # noqa: E501
    finally:
        print("...processing finished.")

    mock_prometheus_client.print_report()


if __name__ == "__main__":
    main()
