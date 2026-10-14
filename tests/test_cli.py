from pricescout.cli import main


def test_cli_scrape_runs_and_changes(mirror, tmp_path, capsys):
    out, db = tmp_path / "out", tmp_path / "db.sqlite"
    code = main(["scrape", "--url", mirror.url, "--max-pages", "1", "--delay", "0",
                 "--out", str(out), "--db", str(db), "--formats", "csv,xlsx"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "products      : 20" in printed
    assert len(list(out.glob("*.csv"))) == 1 and len(list(out.glob("*.xlsx"))) == 1

    assert main(["runs", "--db", str(db)]) == 0
    assert "n_products" in capsys.readouterr().out
    assert main(["changes", "--db", str(db)]) == 0
    assert "No changes" in capsys.readouterr().out


def test_cli_categories(mirror, capsys):
    assert main(["categories", "--url", mirror.url]) == 0
    printed = capsys.readouterr().out
    assert "Poetry" in printed and "8 categories" in printed


def test_cli_unknown_category_exit_code(mirror, tmp_path, capsys):
    code = main(["scrape", "--url", mirror.url, "--category", "Cooking", "--delay", "0",
                 "--out", str(tmp_path), "--db", str(tmp_path / "x.db")])
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_cli_extract_sample_with_structured_data(tmp_path, capsys):
    out = tmp_path / "products.csv"
    assert main(["extract", "--sample", "1", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "method: structured data" in printed and "OK  Aurora ANC Wireless Headphones · 649.00 AED" in printed
    assert out.read_text(encoding="utf-8-sig").splitlines()[0].startswith("title,price,currency")
    assert main(["extract", "--sample", "2"]) == 3  # no structured data and no AI engine
    assert "Choose an AI engine" in capsys.readouterr().out
