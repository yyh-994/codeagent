"""测试 Agent 工具函数 — mock 外部 API"""
import pytest
from unittest.mock import MagicMock, patch


class TestWebSearch:
    def test_search_success(self):
        """mock SerpAPI 返回正常结果。"""
        mock_search = MagicMock()
        mock_search.get_dict.return_value = {
            "organic_results": [
                {"title": "Test", "link": "http://example.com", "snippet": "A test result"},
            ]
        }

        # GoogleSearch 是 lazy import，只存在于 serpapi 模块
        with patch.dict("os.environ", {"SERPAPI_API_KEY": "test-key"}), \
             patch("serpapi.GoogleSearch", return_value=mock_search):
            from back.agent.tools import web_search
            result = web_search.invoke({"query": "test"})

        assert result["success"] is True
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Test"

    def test_search_no_api_key(self):
        """无 API key → 返回 error dict（success=False）。"""
        from back.agent.tools import web_search
        result = web_search.invoke({"query": "test"})
        assert result["success"] is False
        assert "error" in result


class TestStudentTools:
    def test_query_school_db_function(self):
        """验证 query_school_db 存在。"""
        from back.agent.tools import query_school_db
        assert query_school_db is not None


class TestRagTools:
    def test_search_knowledge_base_function(self):
        """验证 search_knowledge_base 存在。"""
        from back.agent.tools import search_knowledge_base
        assert search_knowledge_base is not None

    def test_upload_and_index_pdf_function(self):
        """验证 upload_and_index_pdf 存在。"""
        from back.agent.tools import upload_and_index_pdf
        assert upload_and_index_pdf is not None


class TestImageGenTools:
    def test_generate_image_function(self):
        from back.agent.tools import generate_image
        assert generate_image is not None
