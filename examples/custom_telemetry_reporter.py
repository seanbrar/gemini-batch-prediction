#!/usr/bin/env python3
"""
Minimal custom telemetry reporter example for gemini-batch.
Shows how to print timing and metric events as they happen.
"""

import os

os.environ["GEMINI_TELEMETRY"] = "1"

from gemini_batch import BatchProcessor, TelemetryContext


class PrintReporter:
    def record_timing(self, scope: str, duration: float, **metadata):
        depth = metadata.get("depth", 0)
        indent = "  " * depth
        print(f"[METRIC] {indent}TIMING: scope={scope}, duration={duration:.4f}s")

    def record_metric(self, scope, value, **meta):
        print(f"[metric] {scope}: {value}")

    # # Production extensions (commented out):
    # def __init__(self):
    #     self.timings = []
    #     self.metrics = []
    #
    # def record_timing(self, scope, duration, **meta):
    #     self.timings.append({"scope": scope, "duration": duration, "meta": meta})
    #     print(f"[timing] {scope}: {duration:.3f}s")
    #
    # def record_metric(self, scope, value, **meta):
    #     self.metrics.append({"scope": scope, "value": value, "meta": meta})
    #     print(f"[metric] {scope}: {value}")
    #
    # def export_to_csv(self, filename="telemetry.csv"):
    #     import csv
    #     with open(filename, 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["type", "scope", "value", "meta"])
    #         for t in self.timings:
    #             writer.writerow(["timing", t["scope"], t["duration"], str(t["meta"])])
    #         for m in self.metrics:
    #             writer.writerow(["metric", m["scope"], m["value"], str(m["meta"])])


def main():
    reporter = PrintReporter()
    tele = TelemetryContext(reporter)
    processor = BatchProcessor()
    processor.tele = tele

    content = "Gemini-batch enables efficient batch Q&A."
    questions = ["What does it do?", "How is it efficient?"]

    print("Processing...")
    results = processor.process_questions(content, questions)
    print("Answers:", *results["answers"], sep="\n- ")


if __name__ == "__main__":
    main()
