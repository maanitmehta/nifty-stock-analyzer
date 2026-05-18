"""Tests for nifty_analyzer.universe — NSA-3."""

from __future__ import annotations

import pytest

from nifty_analyzer.universe import get_metadata, get_ticker_list, load_universe, search_stocks


class TestLoadUniverse:
    def test_nifty50_loads_50_stocks(self) -> None:
        df = load_universe("nifty50")
        assert len(df) == 50

    def test_nifty50_has_required_columns(self) -> None:
        df = load_universe("nifty50")
        for col in ("nse_symbol", "company_name", "sector"):
            assert col in df.columns, f"Missing column: {col}"

    def test_nifty50_no_null_symbols(self) -> None:
        df = load_universe("nifty50")
        assert df["nse_symbol"].notna().all()
        assert (df["nse_symbol"] != "").all()

    def test_nifty50_all_marked_in_nifty50(self) -> None:
        df = load_universe("nifty50")
        assert (df["in_nifty50"] == "True").all()

    def test_invalid_filter_raises(self) -> None:
        with pytest.raises((ValueError, FileNotFoundError)):
            load_universe("nifty10")  # type: ignore[arg-type]

    def test_nse_all_missing_raises_file_not_found(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("nifty_analyzer.config.settings.universe_dir", tmp_path)
        with pytest.raises(FileNotFoundError, match="nse_all.csv"):
            load_universe("nse_all")


class TestGetTickerList:
    def test_tickers_have_ns_suffix(self) -> None:
        tickers = get_ticker_list("nifty50")
        assert all(t.endswith(".NS") for t in tickers)

    def test_ticker_count_matches_universe(self) -> None:
        df = load_universe("nifty50")
        tickers = get_ticker_list("nifty50")
        assert len(tickers) == len(df)

    def test_known_ticker_present(self) -> None:
        tickers = get_ticker_list("nifty50")
        assert "RELIANCE.NS" in tickers
        assert "TCS.NS" in tickers
        assert "HDFCBANK.NS" in tickers


class TestGetMetadata:
    def test_known_stock_returns_dict(self) -> None:
        meta = get_metadata("RELIANCE")
        assert isinstance(meta, dict)
        assert meta["company_name"] == "Reliance Industries"
        assert meta["sector"] == "Energy"

    def test_it_stock(self) -> None:
        meta = get_metadata("TCS")
        assert meta["sector"] == "Information Technology"

    def test_unknown_stock_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="FAKESTK999"):
            get_metadata("FAKESTK999")


class TestSearchStocks:
    def test_search_by_symbol(self) -> None:
        results = search_stocks("RELIANCE", filter="nifty50")
        assert len(results) >= 1
        assert "RELIANCE" in results["nse_symbol"].values

    def test_search_by_company_name(self) -> None:
        results = search_stocks("infosys", filter="nifty50")
        assert len(results) >= 1

    def test_search_case_insensitive(self) -> None:
        lower = search_stocks("tata", filter="nifty50")
        upper = search_stocks("TATA", filter="nifty50")
        assert len(lower) == len(upper)

    def test_search_no_results_returns_empty(self) -> None:
        results = search_stocks("ZZZNOMATCH999", filter="nifty50")
        assert results.empty
