# Databricks notebook source
"""Governed ingestion for label-symbol specification documents.

Place DOCX, PDF, or image specifications in a Unity Catalog Volume, then run:

    extracted = extract_symbol_specifications("/Volumes/<catalog>/<schema>/<volume>/symbols")
    display(extracted)  # regulatory/label owner reviews citations and confidence
    publish_approved_catalog(extracted, "/Volumes/<catalog>/<schema>/<volume>/symbol_specifications.json")

``ai_parse_document`` handles the binary document and ``ai_extract`` v2.1
provides cited, confidence-scored fields. The generated JSON is the catalog
consumed by the B300 app after it is reviewed and bundled with a deployment.
"""

import json
from pyspark.sql import DataFrame, SparkSession


SYMBOL_SPECIFICATION_SCHEMA = r'''{
  "symbol_id": {"type": "string", "description": "Unique symbol/artwork identifier, such as 100025. Preserve leading zeros."},
  "symbol_name": {"type": "string", "description": "Official symbol name."},
  "reference": {"type": "string", "description": "Standard, regulation, or controlled-document reference."},
  "required_width_mm": {"type": "number", "description": "Exact required printed width in millimetres; null if not explicitly stated."},
  "required_height_mm": {"type": "number", "description": "Exact required printed height in millimetres; null if not explicitly stated."},
  "minimum_size_mm": {"type": "number", "description": "Explicit minimum printed size in millimetres; null if not stated."},
  "size_specification": {"type": "string", "description": "Exact source wording for size, clear space, scale, or placement."},
  "color_specification": {"type": "string", "description": "Required colours or monochrome requirement."},
  "applicable_for": {"type": "string", "description": "Products, countries, or packaging to which the symbol applies."},
  "usage_notes": {"type": "string", "description": "Mandatory reproduction, legibility, or text-inside-symbol requirements."}
}'''

EXTRACTION_INSTRUCTIONS = """
Extract only facts explicitly stated by this controlled symbol specification.
Do not infer a physical size from artwork pixels. Preserve IDs and units exactly.
Use null for missing numeric values. Return one record per source document.
"""


def extract_symbol_specifications(source_path: str, spark: SparkSession | None = None) -> DataFrame:
    """Parse Volume documents and return a cited, reviewable extraction result."""
    spark = spark or SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("Run this in a Databricks notebook, job, or pipeline.")

    source = spark.read.format("binaryFile").load(source_path)
    source.createOrReplaceTempView("symbol_specification_source")
    return spark.sql(f"""
      WITH parsed AS (
        SELECT path AS source_document,
          ai_parse_document(content, map('version', '2.0', 'descriptionElementTypes', '')) AS parsed
        FROM symbol_specification_source
      ), extracted AS (
        SELECT source_document,
          ai_extract(parsed, '{SYMBOL_SPECIFICATION_SCHEMA}', map(
            'version', '2.1',
            'instructions', '{EXTRACTION_INSTRUCTIONS.strip()}',
            'enableCitations', 'true',
            'enableConfidenceScores', 'true'
          )) AS extraction
        FROM parsed
        WHERE parsed:error_status IS NULL
      )
      SELECT
        source_document,
        extraction:response:symbol_id:value::STRING AS symbol_id,
        extraction:response:symbol_name:value::STRING AS symbol_name,
        extraction:response:reference:value::STRING AS reference,
        extraction:response:required_width_mm:value::DOUBLE AS required_width_mm,
        extraction:response:required_height_mm:value::DOUBLE AS required_height_mm,
        extraction:response:minimum_size_mm:value::DOUBLE AS minimum_size_mm,
        extraction:response:size_specification:value::STRING AS size_specification,
        extraction:response:color_specification:value::STRING AS color_specification,
        extraction:response:applicable_for:value::STRING AS applicable_for,
        extraction:response:usage_notes:value::STRING AS usage_notes,
        extraction:metadata:citations AS citations,
        extraction:metadata AS extraction_metadata,
        extraction:error_message::STRING AS extraction_error
      FROM extracted
    """)


def publish_approved_catalog(reviewed: DataFrame, output_json_path: str) -> None:
    """Publish only human-approved rows to the app's JSON catalog.

    Add an ``approved`` boolean column in the review workflow. This deliberately
    prevents a model extraction from changing label sizing without review.
    """
    if "approved" not in reviewed.columns:
        raise ValueError("Add an approved boolean column after reviewing the extraction results.")
    records = [
        row.asDict(recursive=True)
        for row in reviewed.filter("approved = true AND extraction_error IS NULL").collect()
    ]
    for record in records:
        record["approved"] = True
        record["source_document"] = record.pop("source_document", None)
        record.pop("citations", None)
        record.pop("extraction_metadata", None)
        record.pop("extraction_error", None)
    dbutils.fs.put(output_json_path, json.dumps({"symbols": records}, indent=2), overwrite=True)
