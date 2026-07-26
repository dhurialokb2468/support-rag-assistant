from unittest.mock import MagicMock

from app.query_processor import (
    QueryProcessor,
    extract_query_metadata,
    generate_search_queries,
)


def test_version_extraction():
    processor = QueryProcessor()

    res1 = processor.process("Issue after version 3.2 upgrade")
    assert res1.version == "3.2"

    res2 = processor.process("Running on v3.2.1 in production")
    assert res2.version == "3.2.1"

    res3 = processor.process("InsightFlow 3.2 export bug")
    assert res3.version == "3.2"


def test_error_code_extraction():
    processor = QueryProcessor()

    res1 = processor.process("CSV export failed with EXP-3204")
    assert res1.error_codes == ["EXP-3204"]

    res2 = processor.process("Encountered AUTH-101 and SYNC-2201 errors during sync")
    assert "AUTH-101" in res2.error_codes
    assert "SYNC-2201" in res2.error_codes
    assert len(res2.error_codes) == 2


def test_category_extraction():
    processor = QueryProcessor()

    res_reporting = processor.process("How to export CSV report?")
    assert res_reporting.category == "reporting"

    res_auth = processor.process("Login authentication token failed")
    assert res_auth.category == "authentication"

    res_billing = processor.process("Monthly invoice and subscription payment error")
    assert res_billing.category == "billing"

    res_integrations = processor.process("Salesforce webhook synchronization delay")
    assert res_integrations.category == "integrations"

    res_permissions = processor.process("User role permission denied")
    assert res_permissions.category == "permissions"

    res_user_mgmt = processor.process("User management account provisioning")
    assert res_user_mgmt.category == "user-management"


def test_product_extraction():
    processor = QueryProcessor()

    res1 = processor.process("InsightFlow system status")
    assert res1.product == "InsightFlow"

    res2 = processor.process("How do I clear cache in insightflow?")
    assert res2.product == "InsightFlow"


def test_queries_with_no_metadata():
    processor = QueryProcessor()
    res = processor.process("How do I fix a general problem?")

    assert res.product is None
    assert res.version is None
    assert res.category is None
    assert res.error_codes == []
    assert isinstance(res.extracted_terms, list)


def test_multiple_technical_identifiers():
    query = "InsightFlow 3.2 CSV export fails with EXP-3204 in Salesforce connector"
    res = extract_query_metadata(query)

    assert res.product == "InsightFlow"
    assert res.version == "3.2"
    assert res.error_codes == ["EXP-3204"]
    assert res.category in ["reporting", "integrations"]

    assert "InsightFlow" in res.extracted_terms
    assert "3.2" in res.extracted_terms
    assert "EXP-3204" in res.extracted_terms
    assert "CSV" in res.extracted_terms
    assert "Salesforce" in res.extracted_terms


def test_generate_search_queries_success():
    mock_generator = MagicMock()
    mock_generator.generate_json.return_value = ({
        "queries": [
            "InsightFlow 3.2 CSV export fails with EXP-3204",
            "CSV report export throwing EXP-3204 error on InsightFlow 3.2",
            "InsightFlow 3.2 CSV export configuration guide"
        ]
    }, 0.12)

    original = "InsightFlow 3.2 CSV export fails with EXP-3204"
    queries = generate_search_queries(original, generator=mock_generator, max_queries=3)

    assert len(queries) == 3
    assert queries[0] == original
    assert "EXP-3204" in queries[1]
    assert "InsightFlow 3.2" in queries[2]


def test_generate_search_queries_max_limit_and_original_preservation():
    mock_generator = MagicMock()
    mock_generator.generate_json.return_value = ({
        "queries": [
            "Symptom variation query",
            "Documentation style variation query",
            "Extra variation 1",
            "Extra variation 2"
        ]
    }, 0.05)

    original = "Original standalone search query"
    queries = generate_search_queries(original, generator=mock_generator, max_queries=3)

    assert len(queries) == 3
    assert queries[0] == original
    assert "Symptom variation query" in queries
    assert "Documentation style variation query" in queries


def test_generate_search_queries_fallback_on_error():
    mock_generator = MagicMock()
    mock_generator.generate_json.side_effect = RuntimeError("Ollama service unavailable")

    original = "InsightFlow auth token error AUTH-101"
    queries = generate_search_queries(original, generator=mock_generator)

    assert queries == [original]

