"""deid-battery: a portable, config-driven battery of de-identification systems.

Run many de-id models over the same documents, emit a unified span schema,
optionally inject known metadata (patient/caregiver names) into the rule-based
engines and the post-processor, and evaluate core PII recall + non-PII redaction
rate against a gold bundle.
"""
__version__ = "0.1.0"
