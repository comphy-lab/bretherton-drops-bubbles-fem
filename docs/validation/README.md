# Void-bubble validation report

Scientific report on the sharp-interface void-bubble calculations and their
comparison with reference laws, with published axisymmetric finite-element
results, and with the companion volume-of-fluid campaign.

| File or directory | Contents |
|---|---|
| `main.tex` | Report source |
| `references.bib` | Bibliography |
| `main.pdf` | Compiled report |
| `figures/` | PDF and PNG figures, with their plotting scripts |
| `data/` | Compact CSVs holding the campaign data behind the report's figures and tables; verification-stage values come from `../../receipts/` |
| `Makefile` | LaTeX and figure build |

## Compile

Requires LaTeX with `latexmk`, BibTeX, `natbib`, `graphicx`, `booktabs`,
`amsmath`, `amssymb`, `geometry` and `hyperref`.

From the repository root:

```bash
make -C docs/validation
```

This produces `docs/validation/main.pdf`; auxiliary files remain under
`build/`.

## Regenerate the figures

Requires Python with NumPy and Matplotlib, plus LaTeX and `dvipng` for the
mathematical text. It does **not** require pyoomph or any simulation output:

```bash
make -C docs/validation figures
make -C docs/validation
```

`make figures` uses `python3` by default; point it at an interpreter that has
NumPy and Matplotlib with `make PYTHON=/path/to/python figures`. The figure
PDFs are committed, so an ordinary `make` needs LaTeX only.

## Data

| File | Contents |
|---|---|
| `data/film-sweep.csv` | The 25-point sweep: `Ca_in`, `Ca_b`, `H`, reference values, checks, resolution and route |
| `data/rear-cap.csv` | Rear-cap indentation, `m` and the speed ratio across the sweep |
| `data/tail-onset.csv` | Onset scan for the re-entrant rear cap, at two film resolutions |
| `data/profiles/` | Converged interface profiles, one per decade of `Ca_b` |
| `data/giavedoni-saita-1997.csv` | Points digitised from the figures of Giavedoni & Saita (1997), with the source figure recorded per point |
| `data/companion-basilisk.csv` | The companion volume-of-fluid campaign's film table, reproduced with that campaign's agreement |

The two external series are not produced by this solver. The digitised
literature carries reading error of a few per cent at small capillary number,
which is why it is drawn as a large soft symbol rather than a crisp marker.
The companion table's `outlet_limited` flag marks cases that reached the end of
their laboratory-frame domain; those points are drawn open and excluded from
the summary statistics in the report.
