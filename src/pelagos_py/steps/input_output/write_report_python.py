# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes PDF reports on the current data passed through the pipeline.

The report is built entirely in Python with `fpdf2 <https://py-pdf.github.io/fpdf2/>`_,
so no external toolchain (LaTeX, Sphinx) is required to produce the PDF.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step
import pelagos_py.utils.diagnostics as di

#### Custom imports ####
from fpdf import FPDF
from fpdf.enums import (
    XPos,
    YPos,
    WrapMode,
    MethodReturnValue,
    TableBordersLayout,
    TableCellStyle,
)
from fpdf.fonts import FontFace
from datetime import datetime, timezone
import getpass
import os
import platform
import json
import shutil
import tempfile
import yaml
from importlib.metadata import version, PackageNotFoundError
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm
import numpy as np


#   The core PDF fonts are latin-1 only. Map the symbols we expect to plain
#   text/latin-1 equivalents so they render rather than raising on output.
CHAR_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "σ": "sigma",
    "μ": "u",
    "–": "-",
    "—": "-",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}


def sanitize(text) -> str:
    """Make ``text`` safe for FPDF's latin-1 core fonts.

    Known unicode symbols are swapped for readable equivalents; anything else
    outside latin-1 is replaced so :meth:`FPDF.output` never raises.
    """
    text = str(text)
    for k, v in CHAR_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


#   Project metadata shown on the report.
GITHUB_URL = "https://github.com/NOC-OBG-Autonomy/pelagos-py"

#   Hyperlink colour: a dark teal (rather than the default web blue) so links sit
#   more comfortably against the report's traditional, Times-set look.
_LINK_TEAL = (0, 102, 102)

#   The OG1 format user manual, linked from the Format Checker heading when the
#   compliance checker that ran is the OG1 one.
OG1_MANUAL_URL = "https://github.com/OceanGlidersCommunity/OG-format-user-manual"

#   Compliance-checker codes that we render with a friendlier label and a link
#   to the format's documentation, keyed by the lower-cased checker name.
CC_CHECKER_LABELS = {
    "og": ("OG1", OG1_MANUAL_URL),
    "og1": ("OG1", OG1_MANUAL_URL),
}

#   Placeholder written into ``dataset_id`` when the OG1 file lacks one. Used to
#   suppress the (meaningless) ID label on the QC plots.
UNKNOWN_DATASET_ID = "unknown dataset ID"

#   Human-readable descriptions for the Argo QC flag mnemonics stored in each
#   QC variable's ``flag_meanings`` attribute, used for the end-of-report index.
QC_FLAG_DESCRIPTIONS = {
    "NO_QC": "No QC performed",
    "GOOD": "Good data",
    "PROB_GOOD": "Probably good data",
    "PROB_BAD": "Probably bad data",
    "BAD": "Bad data",
    "VALUE_CHANGED": "Value changed / adjusted",
    "NOT_USED": "Not used",
    "ESTIMATED": "Estimated / interpolated value",
    "MISSING": "Missing value",
}

#   Fallback Argo flag table (value, mnemonic) used when the dataset carries no
#   QC variable with flag_values/flag_meanings attributes to read from.
_DEFAULT_QC_FLAGS = [
    (0, "NO_QC"), (1, "GOOD"), (2, "PROB_GOOD"), (3, "PROB_BAD"), (4, "BAD"),
    (5, "VALUE_CHANGED"), (6, "NOT_USED"), (7, "NOT_USED"), (8, "ESTIMATED"),
    (9, "MISSING"),
]
#   The NOC logo lives in utils/ (alongside the other shared, non-step assets)
#   rather than in a dedicated assets folder.
LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "utils", "noc_logo_square.svg"
)


def pelagos_version() -> str:
    """Return the installed pelagos-py version, or ``unknown``."""
    try:
        return version("pelagos_py")
    except PackageNotFoundError:
        return "unknown"


def long_date(when: datetime) -> str:
    """Format a datetime as e.g. ``23rd June 2026, 22:49 UTC``."""
    day = when.day
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {when.strftime('%B %Y, %H:%M')} UTC"


def current_info() -> dict:
    """Returns current operator information from when the report is being generated."""

    now = datetime.now(timezone.utc)

    info = {
        "timestamp_utc": now.isoformat(),
        "user": getpass.getuser(),
        "version": pelagos_version(),  #   Normally done with __version__.
        "python_version": platform.python_version(),
        "system": f"{platform.system()}: {platform.release()}",
    }

    return info


def build_qc_dict(data: xr.Dataset) -> dict:
    """
    Return a dictionary of all QC variable names and their corresponding QC attributes.

    Can be expanded in the future if additional attributes related to testing are added.
    Tests are ID'd using `_flag_cts` suffix in variable test parameters

    Parameters
    ----------
    data : Xarray DataSet
        The top level data containing all the relevant QC variables.

    Returns
    -------
    qc_dict : dict
        Nested dictionaries of QC variables with test names and results.

        Structure::

            {
                "VAR_QC": {
                    "qc_name": {
                        "params": {...},
                        "flag_counts": {...},
                        "stats": {...},
                    },
                    "qc_name_2": {
                        ...
                    },
                }
            }

    TODO: Move to utils? Does it belong here?
    """
    qc_dict = {}
    for var in data.data_vars:
        if not var.endswith("_QC"):
            continue

        attrs = data[var].attrs
        qc_dict[var] = {}

        # ID tests that were run for indexing.
        # _flag_cts seems like the least standardized name to ID qc with
        qc_names = [
            attr.replace("_flag_cts", "")
            for attr in attrs
            if attr.endswith("_flag_cts")
        ]

        for test in qc_names:
            params_key = f"{test}_params"
            flag_key = f"{test}_flag_cts"
            stats_key = f"{test}_stats"

            # Safely load JSON fields if present (how they were likely saved)
            params = json.loads(attrs[params_key]) if params_key in attrs else {}
            flag_cts = json.loads(attrs[flag_key]) if flag_key in attrs else {}
            stats = json.loads(attrs[stats_key]) if stats_key in attrs else {}

            qc_dict[var][test] = {
                "params": params,
                "flag_counts": flag_cts,
                "stats": stats,
            }

    return qc_dict


def flatten_qc_dict(qc_dict: dict) -> list:
    """
    Flatten QC dictionary into list of table rows.

    Intended for use in the report's QC metrics table.

    Parameters
    ----------
    qc_dict : dict
        Dictionary of QC results.

    Returns
    -------
    rows: list of list
        A list of rows suitable for tabular display. Each row is a list::

            [qc_var, qc_name, flag, formatted_count]

        - `qc_var` : str, the QC variable name
        - `qc_name` : str, the name of the QC test
        - `flag` : str, QC flag value
        - `formatted_count` : str, count formatted with thousands separator
    """
    rows = []

    for qc_var, tests in qc_dict.items():
        if not tests:
            continue

        for qc_name, test_data in tests.items():
            flag_counts = test_data.get("flag_counts", {})

            for flag, count in flag_counts.items():
                if count == 0:
                    continue

                rows.append(
                    [
                        qc_var,
                        qc_name,
                        flag,
                        f"{count:,}",
                    ]
                )

    return rows


### PDF document


class _BooktabsBorders(TableBordersLayout):
    """A clean, scientific-paper table style: horizontal rules only.

    Like LaTeX's ``booktabs``, the only lines drawn are a rule along the top of
    the table, a rule beneath the heading row(s) and a rule along the bottom.
    There are no vertical lines and no rules between body rows, so the table
    reads as a typeset scientific table rather than a basic grid.
    """

    def cell_style_getter(
        self,
        row_idx,
        col_idx,
        col_pos,
        num_heading_rows,
        num_rows,
        num_col_idx,
        num_col_pos,
    ) -> TableCellStyle:
        top = row_idx == 0
        bottom = (row_idx == num_heading_rows - 1) or (row_idx == num_rows - 1)
        return TableCellStyle(left=False, right=False, top=top, bottom=bottom)


#   Shared instance; the layout is stateless so one is enough.
BOOKTABS = _BooktabsBorders()


class _ColumnFlow:
    """Lay content out in newspaper-style columns on a :class:`ReportPDF`.

    Content is placed top-to-bottom down one column, then the next, then onto a
    fresh page. Callers ask for a block of a given height with :meth:`place`,
    which returns the ``(x, y)`` at which to draw it, handling column and page
    breaks. Auto page break is disabled for the flow's lifetime and restored by
    :meth:`finish`.
    """

    def __init__(self, pdf: "ReportPDF", ncols: int = 2, col_gap: float = 6):
        self.pdf = pdf
        self.ncols = ncols
        self.col_gap = col_gap
        self.col_w = (pdf.epw - col_gap * (ncols - 1)) / ncols
        self.top = pdf.get_y()
        self.col = 0
        self.y = self.top
        self.max_y = self.top
        self._saved_apb = pdf.auto_page_break
        pdf.set_auto_page_break(False)

    def _advance(self) -> None:
        """Move to the next column, or a fresh page once columns are used up."""
        self.col += 1
        if self.col >= self.ncols:
            self.pdf.set_auto_page_break(self._saved_apb)
            self.pdf.add_page()
            self.pdf.set_auto_page_break(False)
            self.top = self.pdf.get_y()
            self.col = 0
        self.y = self.top

    def keep_together(self, height: float) -> None:
        """Advance to the next column/page if ``height`` won't fit but could.

        Used to avoid splitting a small block (e.g. a check and its first
        message) across a column boundary when it would fit whole elsewhere.
        """
        capacity = self.pdf.page_break_trigger - self.top
        if (
            self.y > self.top
            and self.y + height > self.pdf.page_break_trigger
            and height <= capacity
        ):
            self._advance()

    def place(self, height: float) -> tuple:
        """Reserve ``height`` in the current column; return the ``(x, y)`` to draw at."""
        if self.y > self.top and self.y + height > self.pdf.page_break_trigger:
            self._advance()
        x = self.pdf.l_margin + self.col * (self.col_w + self.col_gap)
        y = self.y
        self.y += height
        self.max_y = max(self.max_y, self.y)
        return x, y

    def finish(self) -> None:
        """Restore auto page break and move the cursor below the filled columns."""
        self.pdf.set_auto_page_break(self._saved_apb)
        self.pdf.set_y(self.max_y)


class ReportPDF(FPDF):
    """An :class:`fpdf.FPDF` subclass with the report's title page and helpers.

    Pages after the title page carry a running header (the report title) and a
    page-number footer. Heading/body/table/image helpers wrap the lower-level
    FPDF primitives so the section builders below stay terse.
    """

    def __init__(
        self,
        title="Pipeline Report",
        subtitle=None,
        steps=None,
        pipeline_name=None,
        pipeline_description=None,
        track_map_path=None,
    ):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = title
        self.report_subtitle = subtitle
        self.report_steps = steps or []
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description
        self.track_map_path = track_map_path
        #   A roomy bottom margin keeps body content clear of the page-number
        #   footer (which sits ~15 mm from the foot).
        self.set_auto_page_break(auto=True, margin=20)
        self.set_title(sanitize(title))
        #   Table-of-contents entries (title, page number, internal link id),
        #   filled in by :meth:`section_heading` as each section is written.
        self.toc = []

    def header(self) -> None:
        """Running header on every page except the title page."""
        if self.page_no() == 1:
            return
        self.set_font("Times", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, sanitize(self.report_title), align="R")
        self.ln(10)
        self.set_text_color(0)

    def footer(self) -> None:
        """Page-number footer on every page except the title page."""
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)

    def title_page(self) -> None:
        """Write a centred title page (title, subtitle, steps, author, date).

        The whole report uses the Times core font for a traditional,
        scientific (LaTeX-like) look; only verbatim content such as the
        configuration and logfile is set in a monospaced face.
        """
        self.add_page()

        #   NOC logo, centred near the top. Kept small to leave room for the
        #   title-page track map below.
        self.ln(16)
        logo_w = 20
        if os.path.exists(LOGO_PATH):
            try:
                self.image(LOGO_PATH, x=(self.w - logo_w) / 2, w=logo_w)
                self.ln(logo_w + 6)
            except Exception:  # noqa: BLE001 - logo is decorative, never fatal
                self.ln(16)
        else:
            self.ln(16)

        self.set_font("Times", "B", 28)
        self.multi_cell(
            0, 12, sanitize(self.report_title), align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(6)
        if self.report_subtitle:
            self.set_font("Times", "", 16)
            self.multi_cell(
                0, 10, sanitize(self.report_subtitle), align="C",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        #   Pipeline name and description, straight from the configuration.
        if self.pipeline_name:
            self.ln(8)
            self.set_font("Times", "I", 14)
            self.multi_cell(
                0, 8, sanitize(self.pipeline_name), align="C",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        if self.pipeline_description:
            self.set_font("Times", "", 11)
            self.multi_cell(
                0, 6, sanitize(self.pipeline_description), align="C",
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )

        #   Glider track map, centred. Capped in height so the rest of the title
        #   page still fits; omitted silently when no map could be built. The map
        #   is square (1:1), so default to that aspect when it can't be read.
        if self.track_map_path and os.path.exists(self.track_map_path):
            self.ln(6)
            aspect = _image_aspect(self.track_map_path, default=1.0)
            map_w, max_h = 130, 72
            if map_w * aspect > max_h:
                map_w = max_h / aspect
            self.image(self.track_map_path, x=(self.w - map_w) / 2, w=map_w)
            self.ln(2)

        self.ln(8)
        self._steps_abstract()

        #   Provenance: run date, pelagos-py version, runtime environment and a
        #   project link, grouped together near the foot of the title page.
        self.ln(10)
        stamp = long_date(datetime.now(timezone.utc))
        self.set_font("Times", "", 12)
        self.multi_cell(
            0, 8, stamp, align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(2)
        self.set_font("Times", "", 11)
        self.multi_cell(
            0, 6, f"Generated with pelagos-py v{pelagos_version()}", align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.multi_cell(
            0, 6, f"Python {platform.python_version()} on "
            f"{platform.system()} {platform.release()}", align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_font("Times", "U", 11)
        self.set_text_color(*_LINK_TEAL)
        self.multi_cell(
            0, 6, GITHUB_URL, align="C", link=GITHUB_URL,
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.set_text_color(0)

    def _steps_abstract(self) -> None:
        """Render the processing steps as a centred, abstract-like block.

        Indented from both margins and justified, so at a glance the reader
        can see which steps were run (without their parameters).
        """
        if not self.report_steps:
            return

        #   Numbered, arrow-separated sentence reads like a paper abstract.
        listing = "  ".join(
            f"{i}. {name}" for i, name in enumerate(self.report_steps, start=1)
        )

        #   Temporarily narrow the margins to indent the block like an abstract.
        indent = 25
        left, right = self.l_margin, self.r_margin
        self.set_left_margin(left + indent)
        self.set_right_margin(right + indent)
        self.set_x(left + indent)

        self.set_font("Times", "BI", 11)
        self.multi_cell(
            0, 6, "Processing steps", align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        self.ln(1)
        self.set_font("Times", "", 11)
        self.multi_cell(
            0, 5.5, sanitize(listing), align="C",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )

        self.set_left_margin(left)
        self.set_right_margin(right)
        self.set_x(left)

    def h2(self, text: str) -> None:
        """Write a level-2 heading."""
        self.ln(2)
        self.set_font("Times", "B", 16)
        self.multi_cell(0, 9, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def section_heading(self, text: str) -> None:
        """Write a level-2 heading and record it for the contents/index page.

        Captures an internal link to the current page so the closing index can
        list each section with its page number and link straight to it. Call
        once per top-level section, on the page the section begins.
        """
        link = self.add_link()
        self.set_link(link, page=self.page_no())
        self.toc.append((text, self.page_no(), link))
        self.h2(text)

    def contents(self) -> None:
        """Render the recorded sections as a compact, linkable contents list.

        Each row is the section title (a teal internal link to its page) with the
        page number dot-led to the right. Drawn small so the contents stay tidy.
        """
        if not self.toc:
            return
        self.set_font("Times", "", 9)
        page_w = 14  #   narrow right-hand column for the page number
        for title, page, link in self.toc:
            self.set_text_color(*_LINK_TEAL)
            self.cell(self.epw - page_w, 5, sanitize(title), link=link)
            self.set_text_color(0)
            self.cell(
                page_w, 5, str(page), align="R", link=link,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
        self.ln(2)

    def h3(self, text: str) -> None:
        """Write a level-3 heading."""
        self.ln(1)
        self.set_font("Times", "B", 12)
        self.multi_cell(0, 7, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def body(self, text: str, align: str = "LEFT") -> None:
        """Write a paragraph of body text."""
        self.set_font("Times", "", 11)
        self.multi_cell(
            0, 6, sanitize(text), align=align, new_x=XPos.LMARGIN, new_y=YPos.NEXT
        )
        self.ln(2)

    def code_block(self, text: str) -> None:
        """Write monospaced, verbatim text."""
        self.set_font("Courier", "", 8)
        self.multi_cell(0, 4, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def code_listing(
        self, text: str, font_size: int = 8, ncols: int = 1, col_gap: float = 6
    ) -> None:
        """Render verbatim text as a shaded, monospaced code listing.

        Each line keeps a light grey background so the block reads like a
        source-code listing. The text is emitted verbatim (one PDF line per
        source line) so a copy-paste reproduces it faithfully.

        With ``ncols > 1`` the lines flow newspaper-style: top-to-bottom down
        the first column, then the next, then onto a new page. This packs a
        long-but-narrow listing (e.g. a YAML config) into far fewer pages.

        Parameters
        ----------
        text : str
            The verbatim text to render, one source line per line.
        font_size : int
            Monospace font size in points.
        ncols : int
            Number of columns to flow the listing across.
        col_gap : float
            Horizontal gap between columns, in millimetres.
        """
        self.set_font("Courier", "", font_size)
        line_h = font_size * 0.5
        self.set_fill_color(244, 244, 244)
        lines = text.splitlines() or [""]

        if ncols <= 1:
            #   Emit each line verbatim (no added characters) so a copy-paste
            #   reproduces valid YAML. One PDF line per source line; the text is
            #   wrapped narrower than the text width so no line wraps visually.
            for line in lines:
                self.multi_cell(
                    0, line_h, sanitize(line),
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                    fill=True, border=0,
                )
            self.set_fill_color(255, 255, 255)
            self.ln(2)
            return

        #   Newspaper-style columns. Each source line is one fixed-width cell;
        #   long, unbreakable tokens (e.g. a file path) are character-wrapped so
        #   they stay inside the column rather than spilling past it.
        cols = _ColumnFlow(self, ncols=ncols, col_gap=col_gap)
        for line in lines:
            text = sanitize(line) or " "
            h = self.multi_cell(
                cols.col_w, line_h, text,
                border=0, wrapmode=WrapMode.CHAR,
                dry_run=True, output=MethodReturnValue.HEIGHT,
            )
            x, y = cols.place(h)
            self.set_xy(x, y)
            self.multi_cell(
                cols.col_w, line_h, text,
                new_x=XPos.LMARGIN, new_y=YPos.TOP,
                fill=True, border=0, wrapmode=WrapMode.CHAR,
            )
        cols.finish()
        self.set_fill_color(255, 255, 255)
        self.ln(2)

    #   Terminal palette: dark background with light text; log levels are
    #   tinted as they would be in a colourised console.
    _TERMINAL_BG = (24, 24, 27)
    _TERMINAL_FG = (220, 220, 220)
    _TERMINAL_LEVEL_COLORS = {
        "DEBUG": (130, 170, 255),
        "INFO": (180, 220, 180),
        "WARNING": (240, 200, 110),
        "ERROR": (240, 130, 130),
        "CRITICAL": (240, 130, 130),
    }

    def terminal_block(self, lines, font_size: float = 6.5) -> None:
        """Render log lines as a compact, terminal-styled block.

        Each entry is ``(time, level, location, message)``. Lines are set in a
        dark, monospaced panel and tinted by log level, so it reads like
        colourised console output.
        """
        if not lines:
            return
        line_h = font_size * 0.62
        self.set_font("Courier", "", font_size)
        self.set_fill_color(*self._TERMINAL_BG)
        self.set_draw_color(*self._TERMINAL_BG)

        for time_s, level, location, message in lines:
            text = f"{time_s}  {level:<7} {location}: {message}"
            color = self._TERMINAL_LEVEL_COLORS.get(level.upper(), self._TERMINAL_FG)
            self.set_text_color(*color)
            #   multi_cell fills every wrapped line, so the dark panel stays
            #   continuous even when a long message spills onto another line.
            self.multi_cell(
                0, line_h, sanitize(text),
                new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True, border=0,
            )

        self.set_text_color(0)
        self.set_fill_color(255, 255, 255)
        self.ln(2)

    def add_table(
        self, headers, rows, widths=None, font_size=8, align="LEFT", font="Times"
    ) -> None:
        """Render a table with a bold header row.

        Parameters
        ----------
        headers : sequence of str
            Column headings.
        rows : sequence of sequence
            Table body; each inner sequence is one row of cells.
        widths : sequence of number, optional
            Relative column widths. Defaults to equal columns.
        font_size : int
            Body font size in points.
        align : str
            Horizontal cell alignment ("LEFT", "CENTER", "RIGHT").
        font : str
            Font family for the table; "Times" for prose tables, "Courier"
            for verbatim/console-style content such as the logfile.
        """
        self.set_font(font, "", font_size)
        with self.table(
            col_widths=widths,
            text_align=align,
            #   Booktabs-style horizontal rules only (no grid), so the table
            #   reads like a typeset scientific table. A little padding gives
            #   the columns room to breathe without vertical separators.
            borders_layout=BOOKTABS,
            headings_style=FontFace(emphasis="BOLD"),
            padding=(1, 2, 1, 2),
            line_height=font_size * 0.55,
        ) as table:
            table.row([sanitize(h) for h in headers])
            for r in rows:
                table.row([sanitize(c) for c in r])
        self.ln(2)

    def cc_heading(self, label: str, url: str = None, score_text: str = None) -> None:
        """Write a compliance-checker heading, optionally a link to its format docs.

        Parameters
        ----------
        label : str
            The checker label (e.g. ``OG1`` or the raw checker name).
        url : str, optional
            When given, the label is rendered as a blue, underlined hyperlink to
            the format's documentation.
        score_text : str, optional
            A secondary line under the heading (e.g. the compliance score).
        """
        self.ln(2)
        self.set_font("Times", "BU" if url else "B", 13)
        if url:
            self.set_text_color(*_LINK_TEAL)
            self.multi_cell(
                0, 8, sanitize(label), link=url,
                new_x=XPos.LMARGIN, new_y=YPos.NEXT,
            )
            self.set_text_color(0)
        else:
            self.multi_cell(
                0, 8, sanitize(label), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
        if score_text:
            self.set_font("Times", "I", 10)
            self.set_text_color(90)
            self.multi_cell(
                0, 6, sanitize(score_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )
            self.set_text_color(0)
        self.ln(1)

    def cc_checks(self, blocks, ncols: int = 1, col_gap: float = 6) -> None:
        """List compliance-checker findings as a heading-led list.

        Each block is ``(check_name, [messages])``: the check name is set as a
        bold sub-heading and its messages follow as a short list beneath it.
        Blocks flow newspaper-style across ``ncols`` columns; a single full-width
        column suits the (now de-duplicated, compact) findings, whose long
        messages would wrap awkwardly in a narrow column.

        Parameters
        ----------
        blocks : sequence of (str, sequence of str)
            Each ``(check name, messages)`` pair to render.
        ncols : int
            Number of columns to flow the list across.
        col_gap : float
            Horizontal gap between columns, in millimetres.
        """
        if not blocks:
            return

        #   Set small: the Format Checker can list many findings, so a compact
        #   size keeps the section from sprawling across pages.
        title_size, msg_size = 8, 7.5
        title_h, msg_h = 4, 3.4
        cols = _ColumnFlow(self, ncols=ncols, col_gap=col_gap)
        w = cols.col_w
        for name, msgs in blocks:
            #   Measure first so a check stays with (at least) its first message
            #   rather than its heading being orphaned at a column foot.
            self.set_font("Times", "B", title_size)
            h_title = self.multi_cell(
                w, title_h, sanitize(name), border=0,
                dry_run=True, output=MethodReturnValue.HEIGHT,
            )
            self.set_font("Times", "", msg_size)
            msg_heights = [
                self.multi_cell(
                    w, msg_h, sanitize(f"- {m}"), border=0,
                    dry_run=True, output=MethodReturnValue.HEIGHT,
                )
                for m in msgs
            ]
            cols.keep_together(h_title + (msg_heights[0] if msg_heights else 0))

            x, y = cols.place(h_title)
            self.set_xy(x, y)
            self.set_font("Times", "B", title_size)
            self.multi_cell(
                w, title_h, sanitize(name), border=0,
                new_x=XPos.LMARGIN, new_y=YPos.TOP,
            )

            self.set_font("Times", "", msg_size)
            for m, hm in zip(msgs, msg_heights):
                x, y = cols.place(hm)
                self.set_xy(x, y)
                self.multi_cell(
                    w, msg_h, sanitize(f"- {m}"), border=0,
                    new_x=XPos.LMARGIN, new_y=YPos.TOP,
                )
            cols.place(2)  #   small gap after each block
        cols.finish()
        self.ln(2)

    def image_full(self, path: str, aspect: float) -> None:
        """Place an image spanning the full text width, breaking the page if needed.

        Parameters
        ----------
        path : str
            Path to the image file.
        aspect : float
            Image height / width ratio, used to reserve vertical space.
        """
        w = self.epw
        h = w * aspect
        if self.get_y() + h > self.page_break_trigger:
            self.add_page()
        self.image(path, w=w)
        self.ln(4)

    def image_fit(self, path: str, aspect: float, max_h: float) -> None:
        """Place an image spanning the text width but capped at ``max_h`` height.

        When the natural full-width height exceeds ``max_h`` the image is scaled
        down (and centred) so it never grows taller than ``max_h``. This lets
        several plots share a page rather than each taking a whole one. Breaks
        the page first if the image would not fit in the remaining space.

        Parameters
        ----------
        path : str
            Path to the image file.
        aspect : float
            Image height / width ratio.
        max_h : float
            Maximum image height in millimetres.
        """
        w = self.epw
        h = w * aspect
        if h > max_h:
            h = max_h
            w = h / aspect
        if self.get_y() + h > self.page_break_trigger:
            self.add_page()
        #   Centre the (possibly narrowed) image within the text width.
        x = (self.w - w) / 2
        self.image(path, x=x, w=w)
        self.ln(4)


### Section builders


def config_to_yaml(config: dict, width: int = 88) -> str:
    """Serialise the run configuration to compact, comment-free YAML.

    Rebuilt from the parsed configuration, so the result has no comments or
    blank lines and is valid, paste-ready YAML.

    Parameters
    ----------
    config : dict
        The full run configuration.
    width : int
        Column at which long scalar values are wrapped. Set narrow so the
        listing fits a single column when laid out in multiple columns.
    """
    return yaml.safe_dump(
        config,
        sort_keys=False,
        default_flow_style=False,
        width=width,
    ).strip()


def config_section(
    pdf: ReportPDF, config: dict, ncols: int = 2, font_size: int = 8, col_gap: float = 6
) -> None:
    """Write the run configuration as a multi-column YAML code listing.

    The YAML is short-but-tall, so it is flowed across ``ncols`` columns to
    keep it to as few pages as possible. The YAML's wrap width is derived from
    the resulting column width so its lines fit a column without spilling.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    config : dict
        The full run configuration (``pipeline`` block and ``steps`` list).
    ncols : int
        Number of columns to flow the YAML across.
    font_size : int
        Monospace font size for the listing, in points.
    col_gap : float
        Horizontal gap between columns, in millimetres.
    """
    #   Begin on a fresh page like the report's other sections.
    pdf.add_page()
    pdf.section_heading("Configuration")
    if not config:
        pdf.body("No configuration available.")
        return
    pdf.body(
        "The pipeline was run with the following configuration "
        "(comments and blank lines removed):"
    )
    #   Wrap the YAML to (roughly) the number of monospace characters that fit a
    #   single column, so lines don't spill past their column. Courier (a core
    #   font) is 0.6 em wide per character; epw is the usable text width in mm.
    col_w_mm = (pdf.epw - col_gap * (ncols - 1)) / ncols
    char_w_mm = 0.6 * font_size / 72 * 25.4
    chars = max(20, int(col_w_mm / char_w_mm) - 1)
    pdf.code_listing(
        config_to_yaml(config, width=chars),
        font_size=font_size,
        ncols=ncols,
        col_gap=col_gap,
    )


def _image_aspect(path: str, default: float = 0.6) -> float:
    """Return an image's height/width ratio for :meth:`ReportPDF.image_full`.

    Read with matplotlib so no extra dependency is needed; falls back to
    ``default`` if the image cannot be read.
    """
    try:
        arr = plt.imread(path)
        height, width = arr.shape[0], arr.shape[1]
        return height / width
    except Exception:  # noqa: BLE001 - sizing is best-effort, never fatal
        return default


def diagnostics_section(pdf: ReportPDF, captured: list) -> None:
    """Embed the diagnostic plot(s) captured for each step.

    The plots are generated by the pipeline in the background while each step
    runs (regardless of that step's own ``diagnostics`` setting), so this section
    mirrors what the user would see had diagnostics been enabled throughout.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    captured : list of dict
        Per-step records ``{"step": name, "images": [path, ...]}`` in run order,
        as collected by the pipeline. ``None`` or empty when no plots were
        captured (e.g. the pipeline was not run via :meth:`Pipeline.run`).
    """
    if not captured:
        return

    #   Start on a fresh page so the heading isn't orphaned at the foot of the
    #   previous section with its plots breaking onto the next page.
    pdf.add_page()
    pdf.section_heading("Step diagnostics")
    pdf.body(
        "Diagnostic plots generated for each step that produces one. These are "
        "generated for the report regardless of each step's diagnostics setting."
    )
    #   Cap each plot's height so two (heading + plot) fit on a page rather than
    #   each diagnostic taking a whole page.
    max_h = 105
    for entry in captured:
        images = entry.get("images", [])
        if not images:
            continue
        pdf.h3(entry.get("step", "Step"))
        for img in images:
            pdf.image_fit(img, aspect=_image_aspect(img), max_h=max_h)


def qc_section(pdf: ReportPDF, data: xr.Dataset) -> None:
    """
    Write the Quality Control summary table.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.
    """
    #   Start on a fresh page so the heading begins a page rather than trailing
    #   the previous section.
    pdf.add_page()
    pdf.section_heading("Quality Control Summary")

    qc_dict = build_qc_dict(data)
    rows = flatten_qc_dict(qc_dict)

    if not rows:
        pdf.body("No QC tests found.")
        return

    pdf.add_table(
        ["QC Variable", "Test", "Flag", "Count"],
        rows,
        widths=(30, 30, 20, 20),
    )


def add_log(logfile, pdf: ReportPDF, ncols: int = 4) -> None:
    """
    Add and format the logfile as a terminal-style block.

    Note: Requires a designated log_file be initialized in the global pipeline
    configuration parameters.

    Parameters
    ----------
    logfile : str or Path
        Path to the logfile to read.
    pdf : ReportPDF
        The active PDF document being written to.
    ncols : int
        Number of " - " separated columns expected per log line.
    """
    pdf.add_page()
    pdf.section_heading("Logfile of run")

    rows = []
    try:
        with open(logfile, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" - ", maxsplit=ncols - 1)
                if len(parts) != ncols:
                    # malformed or unexpected line, skip
                    continue

                timestamp, level, location, message = parts
                #   Remove date for more space for comment
                timestamp = timestamp.split(" ")[1] if " " in timestamp else timestamp
                location = location.removeprefix("pelagos_py.")

                rows.append((timestamp, level, location, message))
    except FileNotFoundError:
        pdf.body("Logfile not found.")
        return

    if not rows:
        pdf.body("No log entries found.")
        return

    pdf.terminal_block(rows)


def _collapse_repeated_msgs(msgs) -> list:
    """Combine messages that differ only by variable name into a single line.

    Compliance checkers often emit the same finding once per variable (e.g.
    ``"variable TEMP vocabulary ... contains https. Vocabulary URIs use http"``).
    Messages of the shape ``variable <NAME> <rest>`` are grouped by their text
    after the variable name and merged into one line listing every affected
    variable, turning a long, repetitive block into a few lines. Order of first
    appearance is preserved, and any message not of that shape is kept verbatim.
    """
    order = []  # group keys, in first-seen order
    groups = {}
    for m in msgs:
        parts = m.split(maxsplit=2)
        if len(parts) == 3 and parts[0].lower() == "variable":
            #   Group by the text after the variable name (parts[2]).
            key = ("var", parts[2])
            if key not in groups:
                groups[key] = {"rest": parts[2], "vars": []}
                order.append(key)
            groups[key]["vars"].append(parts[1])
        else:
            #   Not "variable <NAME> <rest>": keep verbatim, never merged.
            key = ("single", len(order))
            groups[key] = {"msg": m}
            order.append(key)

    out = []
    for key in order:
        group = groups[key]
        if "msg" in group:
            out.append(group["msg"])
        elif len(group["vars"]) == 1:
            out.append(f"variable {group['vars'][0]} {group['rest']}")
        else:
            out.append(f"variables {', '.join(group['vars'])} {group['rest']}")
    return out


def _render_cc_results(pdf: ReportPDF, cc_data: dict) -> None:
    """Render compliance-checker results as per-checker scores and message tables.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    cc_data : dict
        Mapping of checker name -> result dict (as produced by the compliance
        checker's ``dict_output``), each with ``scored_points``,
        ``possible_points`` and ``all_priorities``.
    """
    for cname, test_data in cc_data.items():
        scored = test_data.get("scored_points")
        possible = test_data.get("possible_points")

        #   Friendly label + docs link for known formats (e.g. OG1); otherwise
        #   fall back to the raw checker name with no link.
        label, url = CC_CHECKER_LABELS.get(str(cname).lower(), (cname, None))
        score_text = None
        if scored is not None and possible is not None:
            score_text = f"Compliance score: {scored}/{possible}"
        pdf.cc_heading(label, url, score_text)

        #   Each failing check becomes a heading-led block (name + its messages)
        #   rather than table rows with a mostly-empty name column.
        blocks = [
            (entry.get("name", "Unknown"), _collapse_repeated_msgs(entry["msgs"]))
            for entry in test_data.get("all_priorities", [])
            if entry.get("msgs")
        ]

        if blocks:
            pdf.cc_checks(blocks)
        else:
            pdf.body("No issues reported.")


def format_checker_section(
    pdf: ReportPDF, cc_results: dict = None, ccfile=None
) -> None:
    """
    Write the Format Checker (compliance checker) results section.

    Populated whenever the Format Checker step ran, regardless of whether a
    report file was saved: the structured results are read from ``cc_results``
    (stashed in the context by the step). When those are unavailable but a saved
    ``ccfile`` exists it is used as a fallback (JSON parsed, any other format
    embedded verbatim).

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    cc_results : dict, optional
        Mapping of checker name -> result dict, as stashed by the Format Checker
        step in ``context["cc_results"]``.
    ccfile : str, optional
        Path to a saved compliance-checker report, used only as a fallback when
        ``cc_results`` is not available.
    """
    pdf.add_page()
    pdf.section_heading("Format Checker results")

    if cc_results:
        _render_cc_results(pdf, cc_results)
    elif ccfile and str(ccfile).endswith(".json"):
        with open(ccfile, mode="r") as f:
            _render_cc_results(pdf, json.load(f))
    elif ccfile:
        with open(ccfile, "r") as f:
            content = f.read()
        pdf.code_block(content)
    else:
        pdf.body("Format Checker ran but produced no detailed results.")


def qc_flag_glossary_rows(data: xr.Dataset) -> list:
    """Build ``[flag, name, meaning]`` rows for the QC flag glossary.

    Reads ``flag_values``/``flag_meanings`` from the first QC variable that
    carries them so the glossary matches the data, falling back to the Argo
    standard table when none is present.

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.

    Returns
    -------
    list of list
        Rows ``[flag_value, mnemonic, human-readable meaning]``.
    """
    values, names = None, None
    for var in data.data_vars:
        attrs = data[var].attrs
        if "flag_values" in attrs and "flag_meanings" in attrs:
            values = list(attrs["flag_values"])
            #   flag_meanings is a comma- and/or space-separated string of codes.
            names = str(attrs["flag_meanings"]).replace(",", " ").split()
            break

    if not values or not names or len(values) != len(names):
        values = [v for v, _ in _DEFAULT_QC_FLAGS]
        names = [n for _, n in _DEFAULT_QC_FLAGS]

    return [
        [str(value), name, QC_FLAG_DESCRIPTIONS.get(name, name)]
        for value, name in zip(values, names)
    ]


def variable_index_rows(data: xr.Dataset) -> list:
    """Build ``[variable, long name, units]`` rows for the variable index.

    The long name falls back to the ``description`` attribute, then to a blank.
    Units are blanked when absent or explicitly "None".

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.

    Returns
    -------
    list of list
        Rows ``[variable_name, long name/description, units]``.
    """
    rows = []
    for var in data.data_vars:
        attrs = data[var].attrs
        long_name = attrs.get("long_name") or attrs.get("description") or ""
        units = attrs.get("units")
        units = "" if not units or str(units).lower() == "none" else str(units)
        rows.append([var, long_name, units])
    return rows


def index_section(pdf: ReportPDF, data: xr.Dataset) -> None:
    """Write the closing index page.

    Repeats the pelagos-py credit, then a linked contents list (each section with
    its page number) followed by the report's reference tables: a QC flag
    glossary, an index of every variable (with long name and units) and the
    glider/mission global attributes. Compact tables so the reference fits on as
    few pages as possible.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.
    """
    pdf.add_page()

    #   Repeat the pelagos-py credit and project link at the very end.
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(
        0, 6, f"Generated with pelagos-py v{pelagos_version()}", align="C",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_font("Times", "U", 11)
    pdf.set_text_color(*_LINK_TEAL)
    pdf.multi_cell(
        0, 6, GITHUB_URL, align="C", link=GITHUB_URL,
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    pdf.set_text_color(0)
    pdf.ln(8)

    pdf.section_heading("Index")

    #   Contents: each report section with its page number, linked in the PDF.
    pdf.h3("Contents")
    pdf.contents()

    #   QC flag glossary: translate the flag values used throughout the report.
    pdf.h3("QC flag glossary")
    pdf.body(
        "Quality-control flags follow the Argo reference table. The table below "
        "translates each flag value used in the QC summary and plots."
    )
    pdf.add_table(
        ["Flag", "Name", "Meaning"],
        qc_flag_glossary_rows(data),
        widths=(15, 40, 65),
        font_size=7,
    )

    #   Variable index: every dataset variable with its long name and units.
    pdf.h3("Variables")
    pdf.add_table(
        ["Variable", "Long name", "Units"],
        variable_index_rows(data),
        widths=(30, 75, 15),
        font_size=7,
    )

    #   Glider/mission information, from the dataset's global attributes.
    glatters = data.attrs
    if "platform_vocabulary" in glatters:  #   May not be in every dataset
        pdf.h3("Glider information")
        pdf.add_table(
            ["Field", "Value"],
            [[key, value] for key, value in glatters.items()],
            widths=(35, 75),
            font_size=7,
        )


### Plot builders (save a figure, return its path)


#   Title-page map palette, echoing the "globe" web view: a muted navy ocean,
#   slate land, faint graticule, and a gold track that brightens from the oldest
#   fix to the newest. Kept a mid-tone (rather than near-black) so the map reads
#   as a softer, lighter panel while the gold track still stands out.
_MAP_OCEAN = "#2b3a57"
_MAP_LAND = "#45526d"
_MAP_COAST = "#7889ad"
_MAP_GRID = "#8294b6"
_MAP_GRID_TEXT = "#d2dcee"
_MAP_GOLD_STOPS = ["#4a3c10", "#b8922a", "#ffd700", "#fff4bf"]
_MAP_START = "#9fe0a0"


#   Cross-section panels (PRES vs TIME, coloured by a variable). Each panel
#   names the variable to colour by (first match wins, so OG1/lower-case and
#   BBP700/BBP532 fallbacks can be listed in preference order) and the exact
#   colourmap stops (low value -> high value; not reversed) used to build a
#   LinearSegmentedColormap. ``special`` flags BBP, which is drawn largest-on-top
#   with smaller markers so its sharp spikes surface rather than hide behind
#   ordinary points.
_CROSS_SECTION_PANELS = (
    {
        "label": "Temperature",
        "candidates": ("TEMP", "TEMPERATURE", "temp"),
        "stops": [
            "#1b1c6e", "#365292", "#5286b7", "#8db4c4", "#dbe5cd",
            "#f1d8b4", "#d9997e", "#c05e4c", "#a0372b", "#811910",
        ],
    },
    {
        "label": "Salinity",
        "candidates": ("PRAC_SALINITY", "ABS_SALINITY", "PSAL", "SALINITY", "salinity"),
        "stops": [
            "#f9e8b1", "#f1c38f", "#e8a074", "#db7c5f", "#cb5c58",
            "#b2425c", "#943061", "#732460", "#511c53", "#321340",
        ],
    },
    {
        "label": "Density",
        "candidates": ("DENSITY", "density", "SIGMA0", "SIGMA_THETA", "POTDENS"),
        "stops": [
            "#e6f1f7", "#c4d8e8", "#a6bed9", "#8ba4c9", "#7489b8", "#636c9f",
            "#595388", "#55406e", "#4e3055", "#3f2040", "#2e1226",
        ],
    },
    {
        "label": "Oxygen",
        "candidates": ("MOLAR_DOXY", "molar_doxy", "DOXY", "molar_deoxy"),
        "stops": [
            "#400000", "#5c0000", "#780000", "#808080", "#8c8c8c", "#979797",
            "#a3a3a3", "#aeaeae", "#baba10", "#dbdb0a", "#fdfd00",
        ],
    },
    {
        "label": "Chlorophyll",
        "candidates": ("CHLA_ADJUSTED", "CHLA", "chla_adjusted", "CHLOROPHYLL"),
        "stops": [
            "#182548", "#2c5398", "#4a88a4", "#87b8b5", "#dae4da",
            "#e6d992", "#a8ab3e", "#56872e", "#285932", "#1b2617",
        ],
    },
    {
        "label": "Backscatter",
        "candidates": ("BBP700", "BBP532", "BBP", "bbp"),
        "special": "bbp",
        "stops": [
            "#cccccc", "#737373", "#000000", "#1335f5", "#3f8df7",
            "#67dffb", "#a1fc4e", "#f8d748", "#ef8733", "#ea3323",
        ],
    },
)

#   Names tried (in order) for the cross-section's shared time and depth axes.
_CS_TIME_CANDIDATES = ("TIME", "time")
_CS_PRES_CANDIDATES = ("PRES", "PRESSURE", "pres", "DEPTH")

#   Default scatter marker size; BBP uses half this so its surfaced high values
#   sit cleanly over the rest rather than smearing.
_CS_MARKER_SIZE = 8.0

#   Only points whose QC flag is one of these are plotted in the cross-sections
#   (good / probably-good / value-changed / interpolated). Points carrying any
#   other flag (e.g. bad, not-used, missing) are masked out before plotting.
_CS_ALLOWED_QC_FLAGS = (0, 1, 2, 5, 8)

#   Gliders log millions of measurements; a million-point scatter renders slowly
#   and bloats the PDF. Thin (consistently across panels, so points stay aligned)
#   to this cap, which still looks dense on an A4 page.
_CS_MAX_POINTS = 120_000


def _first_present(data: xr.Dataset, names) -> str:
    """Return the first of ``names`` that is a variable in ``data``, else ``None``."""
    for name in names:
        if name in data.variables:
            return name
    return None


def _cs_date_format(span_days: float) -> str:
    """Pick a date format string for the cross-section's shared X axis.

    Adapts to the visible span: sub-minute shows seconds; up to a day shows
    hours/minutes; days show the date; longer spans coarsen to month, then year.
    """
    if span_days < 1.0 / 1440.0:      # sub-minute
        return "%H:%M:%S"
    if span_days < 1.0:               # minutes / hours
        return "%H:%M"
    if span_days < 60.0:              # days
        return "%d %b %Y"
    if span_days < 730.0:             # months
        return "%b %Y"
    return "%Y"                       # years


def _var_label(data: xr.Dataset, var: str, label: str) -> str:
    """``label [units]`` for a variable, dropping units when absent or "None"."""
    units = data[var].attrs.get("units")
    if units and str(units).lower() != "none":
        return f"{label} [{units}]"
    return label


def _find_lonlat(data: xr.Dataset):
    """Return ``(lon, lat)`` arrays from the dataset, or ``(None, None)``.

    Looks for the usual OG1 ``LONGITUDE``/``LATITUDE`` names first, then common
    lower-case fallbacks.
    """
    for lon_name, lat_name in (
        ("LONGITUDE", "LATITUDE"),
        ("longitude", "latitude"),
        ("lon", "lat"),
    ):
        if lon_name in data.variables and lat_name in data.variables:
            return data[lon_name].values, data[lat_name].values
    return None, None


def glider_track_map(data: xr.Dataset, outdir: str, ext: str = ".png") -> str:
    """Render the glider track as a dark, web-style map for the title page.

    Mirrors the look of the interactive globe view: a deep-navy ocean with slate
    land, and the track drawn as a gold line that fades from faint (oldest fix)
    to bright (newest), ending in a red "live position" marker. The basemap uses
    cartopy's Natural Earth land/coastline; if those can't be fetched the track
    is still drawn on the navy background so the section never fails or looks
    broken. Returns the saved image path, or ``None`` when no usable track
    exists (no coordinates, or cartopy unavailable).

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.
    outdir : str
        Directory (with trailing separator) to write the figure to.
    ext : str
        Image filetype extension (.png, .svg, etc.).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap
    except Exception:  # noqa: BLE001 - cartopy is optional; skip the map if absent
        return None

    lon, lat = _find_lonlat(data)
    if lon is None:
        return None

    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    valid = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[valid], lat[valid]
    if lon.size < 2:
        return None

    #   Gliders log millions of fixes; thin to a few thousand so the line
    #   collection stays light without changing the track's shape.
    max_pts = 3000
    if lon.size > max_pts:
        stride = int(np.ceil(lon.size / max_pts))
        lon, lat = lon[::stride], lat[::stride]

    #   Padded extent centred on the track.
    lon_mid = 0.5 * (np.nanmin(lon) + np.nanmax(lon))
    lat_mid = 0.5 * (np.nanmin(lat) + np.nanmax(lat))
    span = max(np.nanmax(lon) - np.nanmin(lon), np.nanmax(lat) - np.nanmin(lat))
    span = max(span, 0.05) * 1.6  #   breathing room (and a floor for short tracks)

    #   One degree of longitude covers cos(latitude) times the ground distance of
    #   one degree of latitude, so an equal-degree box stretches the coastline
    #   east-west at high latitudes (e.g. Iceland looks squashed). Widen the
    #   longitude extent by 1/cos(lat) and stretch the axis by the same factor:
    #   the map stays square on the page but the land keeps its true proportions.
    cos_lat = max(np.cos(np.deg2rad(lat_mid)), 0.2)  # floor avoids a near-pole blow-up
    lat_half = span / 2
    lon_half = lat_half / cos_lat
    extent = [lon_mid - lon_half, lon_mid + lon_half,
              lat_mid - lat_half, lat_mid + lat_half]
    #   Finer coastline for tighter views; coarser (and cheaper) when zoomed out.
    scale = "10m" if span < 3 else "50m" if span < 20 else "110m"

    proj = ccrs.PlateCarree()
    #   Square figure (1:1) so the map renders proportionally on the title page.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    fig.patch.set_facecolor(_MAP_OCEAN)
    ax.set_facecolor(_MAP_OCEAN)
    try:
        ax.set_extent(extent, crs=proj)
        ax.set_aspect(1.0 / cos_lat)  #   latitude-correct proportions (see above)
    except Exception:  # noqa: BLE001 - degenerate extents fall back to autoscale
        pass

    #   Land + coastline, with a graceful drop to coarser data (or none) so a
    #   failed Natural Earth download never breaks the report.
    for sc in (scale, "110m"):
        try:
            ax.add_feature(
                cfeature.LAND.with_scale(sc),
                facecolor=_MAP_LAND, edgecolor="none", zorder=0,
            )
            ax.coastlines(resolution=sc, color=_MAP_COAST, linewidth=0.6, zorder=1)
            break
        except Exception:  # noqa: BLE001 - try the next scale, else skip the basemap
            continue

    try:
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.4, color=_MAP_GRID,
            alpha=0.4, linestyle=":",
        )
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {"size": 7, "color": _MAP_GRID_TEXT}
        gl.ylabel_style = {"size": 7, "color": _MAP_GRID_TEXT}
    except Exception:  # noqa: BLE001 - labels are decorative
        pass

    #   Track as a time-faded gold gradient (oldest faint -> newest bright),
    #   with a soft glow underneath so it reads on the dark ocean.
    pts = np.column_stack([lon, lat]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("glidergold", _MAP_GOLD_STOPS)
    glow = LineCollection(
        segs, colors="#ffd70022", linewidth=4.5, transform=proj, zorder=2
    )
    track = LineCollection(segs, cmap=cmap, linewidth=1.8, transform=proj, zorder=3)
    track.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(glow)
    ax.add_collection(track)

    #   The track's brightening gold already shows direction of travel (oldest
    #   faint -> newest bright), so no start/end/position marker is drawn.

    fig.tight_layout(pad=0.3)
    fname = outdir + "glider_track" + ext
    plt.savefig(fname, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return fname


def qc_hist(
    data: xr.Dataset,
    outdir: str,
    var: str,
    dataset_label: str = None,
    xlims: list = [-0.6, 9.6],
    hislim=range(10),
    bins=None,
    ext=".png",
) -> str:
    """
    Create quick quality control histogram figure.

    Left axis:  Quick plot of QC variable's parent
    Right axis: Bins of each flag type, labeled with # of points

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to
    var : str
        The QC variable as listed in `data`
    dataset_label : str, optional
        Text for the figure's shared y-label (the dataset ID). When ``None`` no
        side label is drawn; pass ``None`` for files without a real dataset ID.
    ext : str
        Image filetype extension (.png, .svg, etc.)
    hislim : array-like
        All potential flags of the selected schema (default Argo = 0 to 9, 10 total)
    bins : array-like
        The sequence of bin edges for collection, matching the dimension of hislim
    xlims : list
        Histogram axis bounds. Defaults to Argo (10 flags) with 0.1 padding on each side

    Returns
    -------
    str
        Path to the saved figure.
    """

    var_source = var[:-3]  #   TEMP_QC --> TEMP

    #   Short and wide so three plots fit on a page (see make_plots).
    fig, axs = plt.subplots(ncols=2, figsize=(8, 3.2), layout="constrained")

    #   Prepare the histogram
    ylims = [1, data[var].size]  #   Log axis cannot be 0
    if any(y < 1 for y in ylims):
        raise ValueError
    if bins == None:  #   If not specified, center the bins around each flag integer
        bins = np.arange(len(hislim)) - 0.5

    #   Plot the source variable using xarray.plot for speed.
    #   If all NaN, clarify that on the plot.
    if np.all(np.isnan(data[var_source])):
        axs[0].text(
            0.2, 0.5, f"Data ({var_source}) are NaN", transform=axs[0].transAxes
        )
    else:
        data[var_source].plot(ax=axs[0])
    #   xarray labels the axis with the (often long) description; replace it with
    #   a short name + units so the plot stays uncluttered. Skip the units when
    #   absent or explicitly "None" (e.g. PHASE) so no "[None]" is shown.
    units = data[var_source].attrs.get("units")
    if units and str(units).lower() != "none":
        axs[0].set_ylabel(f"{var_source} [{units}]")
    else:
        axs[0].set_ylabel(var_source)
    axs[0].set_title("")

    if np.all(np.isnan(data[var])):
        axs[1].text(0.2, 0.5, f"Flags ({var}) are NaN", transform=axs[1].transAxes)
    else:
        data[var].plot.hist(
            yscale="log", bins=bins, xticks=hislim, xlim=xlims, ylim=ylims, ax=axs[1]
        )
        bars = axs[1].containers[0]  #   Number of points in each bin
        #   Rotate the counts upright so large numbers don't spill past the bars.
        axs[1].bar_label(bars, fontsize=7, label_type="center", rotation=90)
        axs[1].set_yscale("log")
        #   xarray labels the axis with the flag's (often long) description;
        #   force a short, consistent label instead.
        axs[1].set_xlabel("Quality Flag")
        axs[1].set_title("")

    #   A single title across the multiplot, just the variable name.
    fig.suptitle(var_source)
    #   Only label the side with the dataset ID when a real one is known.
    if dataset_label:
        fig.supylabel(dataset_label)

    fname = outdir + var + ext
    plt.savefig(fname)  #   Save to the outdir
    plt.close(fig)
    return fname


def make_plots(
    pdf: ReportPDF,
    data: xr.Dataset,
    outdir: str,
) -> None:
    """
    Wrapper for plotting glider QC variables quickly.

    There are millions of points per variable, which xarray can plot very quickly
    in specific ways. Here, QC histograms are explored.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to

    TODO: Define long-term storage for this. Is `diagnostics` the right place?
    """
    pdf.add_page()
    pdf.section_heading("QC Plots")

    #   Only plot QC flags that belong to a numeric measurement series. Many
    #   QC variables flag metadata/coordinate fields whose parent is a string,
    #   datetime, or scalar; those cannot be NaN-checked or histogrammed.
    qc_vars = [
        var
        for var in data.data_vars
        if var.endswith("_QC")
        and data[var].ndim >= 1
        and np.issubdtype(data[var].dtype, np.number)
        and var[:-3] in data.data_vars
        and np.issubdtype(data[var[:-3]].dtype, np.number)
    ]

    #   Only label the plots with the dataset ID when a real one is present; the
    #   placeholder set upstream for files without one should not be shown.
    dataset_id = data.attrs.get("dataset_id")
    dataset_label = dataset_id if dataset_id != UNKNOWN_DATASET_ID else None

    for var in tqdm(
        qc_vars,
        colour="green",
        desc=f"\033[97mProgress \033[0m",
        unit="vars",
    ):
        var_source = var[:-3]  #   TEMP_QC --> TEMP
        #   When both the measurement and its flags are entirely NaN there is
        #   nothing to plot. Note it in one line and skip so more useful plots
        #   fit on the page.
        if np.all(np.isnan(data[var_source])) and np.all(np.isnan(data[var])):
            pdf.body(f"{var_source} and {var} are all NaN.", align="C")
            continue
        # Any form of scatter takes ~30 sec, stick with xarray.plot for now (no colorbars, alternative color schemes)
        hist_img = qc_hist(data, outdir, var, dataset_label=dataset_label)
        pdf.image_full(hist_img, aspect=3.2 / 8)


def cross_section_figure(data: xr.Dataset, outdir: str, ext: str = ".png") -> str:
    """Render the A4 cross-section figure (PRES vs TIME, one panel per variable).

    A single static A4-portrait figure: a vertical stack of panels (see
    :data:`_CROSS_SECTION_PANELS`) that all share the TIME X axis. Each panel
    draws PRES (depth, axis inverted so the surface is at the top) against TIME,
    every point a marker coloured by that panel's variable, with a narrow
    vertical-profile strip on the left and a colourbar on the right. Colour
    scaling uses robust percentiles (0.1st / 99.9th) so spikes don't blow out
    the scale, and BBP is drawn largest-on-top.

    Returns the saved image path, or ``None`` when the dataset has no usable
    TIME/PRES coordinates to plot against.

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.
    outdir : str
        Directory (with trailing separator) to write the figure to.
    ext : str
        Image filetype extension (.png, .svg, etc.).
    """
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    time_name = _first_present(data, _CS_TIME_CANDIDATES)
    pres_name = _first_present(data, _CS_PRES_CANDIDATES)
    if time_name is None or pres_name is None:
        return None

    time = np.asarray(data[time_name].values).ravel()
    pres = np.asarray(data[pres_name].values, dtype=float).ravel()
    n = min(time.size, pres.size)
    if n < 2:
        return None
    time, pres = time[:n], pres[:n]

    #   Thin (consistently, so every panel keeps the same points) to a cap that
    #   still looks dense on A4 but renders quickly and keeps the PDF small.
    if n > _CS_MAX_POINTS:
        idx = np.linspace(0, n - 1, _CS_MAX_POINTS).astype(int)
    else:
        idx = np.arange(n)
    time, pres = time[idx], pres[idx]

    #   matplotlib date numbers for the shared X axis (NaT -> NaN, ignored).
    x = mdates.date2num(time)

    panels = _CROSS_SECTION_PANELS
    #   A4-portrait proportions, trimmed a little in height so the "Cross Section
    #   Plots" heading and the figure sit together on one page. constrained_layout
    #   lines up every panel with its left profile strip and right colourbar.
    fig = plt.figure(figsize=(8.27, 10.3), layout="constrained")
    #   Per row: [narrow profile strip] [main cross-section] [thin colourbar].
    gs = fig.add_gridspec(len(panels), 3, width_ratios=[0.22, 1.0, 0.045])

    main_axes = []
    for i, panel in enumerate(panels):
        ax_prof = fig.add_subplot(gs[i, 0])
        #   Main panels share the TIME X axis; profile shares the (inverted) PRES
        #   Y axis with its own main panel.
        ax_main = fig.add_subplot(
            gs[i, 1], sharex=main_axes[0] if main_axes else None, sharey=ax_prof
        )
        cax = fig.add_subplot(gs[i, 2])
        main_axes.append(ax_main)

        cmap = LinearSegmentedColormap.from_list(
            f"xsec_{panel['label'].lower()}", panel["stops"]
        )

        cvar = _first_present(data, panel["candidates"])
        if cvar is None:
            #   Keep the panel (so the layout is fixed at 6 rows) but note the gap.
            ax_main.text(
                0.5, 0.5, f"{panel['label']} not available",
                ha="center", va="center", transform=ax_main.transAxes,
                fontsize=8, color="0.4",
            )
            ax_main.tick_params(labelleft=False)
            ax_prof.tick_params(labelsize=6)
            ax_prof.set_ylabel(_var_label(data, pres_name, pres_name), fontsize=7)
            ax_prof.set_xlabel(panel["label"], fontsize=6)
            cax.axis("off")
            continue

        c = np.asarray(data[cvar].values, dtype=float).ravel()[:n][idx]

        #   Keep only points whose QC flag is in the allowed set; mask the rest to
        #   NaN so they're dropped from both the cross-section and the profile (and
        #   ignored by the percentile colour limits below). Variables without a
        #   ``_QC`` companion are plotted unfiltered.
        qc_name = f"{cvar}_QC"
        if qc_name in data.variables:
            qc = np.asarray(data[qc_name].values, dtype=float).ravel()[:n][idx]
            c = np.where(np.isin(qc, _CS_ALLOWED_QC_FLAGS), c, np.nan)

        #   Robust colour limits: percentiles, not raw min/max, so sharp spikes
        #   don't blow out the scale. Guard the all-NaN case.
        if np.isfinite(c).any():
            vmin = np.nanpercentile(c, 0.1)
            vmax = np.nanpercentile(c, 99.9)
        else:
            vmin, vmax = 0.0, 1.0

        if panel.get("special") == "bbp":
            #   Largest-on-top: sort ascending so the highest BBP values draw last
            #   (NaNs to the bottom), with a smaller marker so spikes surface.
            order = np.argsort(np.where(np.isnan(c), -np.inf, c), kind="stable")
            xo, po, co = x[order], pres[order], c[order]
            size = _CS_MARKER_SIZE / 2
        else:
            xo, po, co = x, pres, c
            size = _CS_MARKER_SIZE

        sc = ax_main.scatter(
            xo, po, c=co, cmap=cmap, vmin=vmin, vmax=vmax, s=size, edgecolors="none"
        )
        #   Left strip: the vertical profile of the same variable (value vs depth).
        ax_prof.scatter(
            c, pres, c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=2, edgecolors="none"
        )

        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(_var_label(data, cvar, panel["label"]), fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        #   PRES axis lives on the profile strip; the main panel reuses it.
        ax_prof.set_ylabel(_var_label(data, pres_name, pres_name), fontsize=7)
        ax_prof.set_xlabel(panel["label"], fontsize=6)
        ax_prof.xaxis.set_major_locator(MaxNLocator(3))
        ax_prof.tick_params(labelsize=6)
        ax_main.tick_params(labelleft=False)

    #   Invert PRES once per row (surface at the top, depth at the bottom). Each
    #   main panel shares Y with its profile strip, so inverting that suffices.
    for ax in main_axes:
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

    #   Shared X is dates: only the bottom panel carries tick labels; upper
    #   panels keep clean axes. The formatter adapts to the visible span.
    span_days = 0.0
    if np.isfinite(x).any():
        span_days = float(np.nanmax(x) - np.nanmin(x))
    for ax in main_axes:
        ax.xaxis_date()
    for ax in main_axes[:-1]:
        ax.tick_params(labelbottom=False)
    bottom = main_axes[-1]
    bottom.xaxis.set_major_locator(mdates.AutoDateLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter(_cs_date_format(span_days)))
    bottom.tick_params(axis="x", labelsize=7)
    bottom.set_xlabel(time_name, fontsize=8)

    fname = outdir + "cross_section" + ext
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def cross_section_section(pdf: ReportPDF, data: xr.Dataset, outdir: str) -> None:
    """Write the Cross Section Plots section (the full-page A4 cross-section figure).

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes.
    outdir : str
        The path to write the figure to.
    """
    pdf.add_page()
    pdf.section_heading("Cross Section Plots")
    img = cross_section_figure(data, outdir)
    if img is None:
        pdf.body("No suitable TIME/PRES data available for cross-section plots.")
        return
    #   Cap the figure to the space left below the heading so the two stay on one
    #   page (the figure is near-A4 height, so a full-width placement would
    #   otherwise spill onto the next page and orphan the heading).
    avail_h = pdf.page_break_trigger - pdf.get_y() - 2
    pdf.image_fit(img, aspect=_image_aspect(img), max_h=avail_h)


@register_step
class WriteDataReportPython(BaseStep):
    """
    Writes a PDF report summarizing the generic plots and statistics of the data.

    Built directly with fpdf2 (no LaTeX/Sphinx toolchain required).

    Base template:
    * Title page (incl. run provenance and pipeline name/description)
    * Cross section plots (PRES vs TIME, coloured per variable)
    * Format Checker results (when that step ran)
    * Configuration
    * Quality control summary
    * Basic plots
    * Logfile
    * Closing index (contents/page index, pelagos-py credit, QC flag glossary, \
variable index, glider information)

    Parameters
    ----------
    title: str
        Name of the report (on title page and header).
    fname: str
        Output .pdf filename; defaults to the filename core when blank.
    delete_figures: bool
        When True (default) the plot images are written to a temporary folder
        and removed once the PDF is built. When False they are kept in a
        uniquely named folder next to the report so successive runs don't
        overwrite each other.
    show_format_check, show_configuration, show_diagnostic_plots, show_qc_summary, \
    show_qc_plots, show_cross_section_plots, show_logs, show_index : bool
        Toggles for the report's sections. All default to True; set any to False
        to omit that section from the report.
    """

    step_name = "Write Data Report (Python)"

    parameter_schema = {
        "title": {
            "type": str,
            "default": None,
            "description": "Report title; defaults to the filename core when blank.",
        },
        "fname": {
            "type": str,
            "default": None,
            "description": "Output .pdf filename; defaults to the filename core when blank.",
        },
        "delete_figures": {
            "type": bool,
            "default": True,
            "description": (
                "Delete the generated plot images after the PDF is built. "
                "When False, keep them in a uniquely named folder beside the report."
            ),
        },
        "show_format_check": {
            "type": bool,
            "default": True,
            "description": "Include the Format Checker results section (when that step ran).",
        },
        "show_configuration": {
            "type": bool,
            "default": True,
            "description": "Include the run configuration (YAML) section.",
        },
        "show_diagnostic_plots": {
            "type": bool,
            "default": True,
            "description": "Include the per-step diagnostic plots section.",
        },
        "show_qc_summary": {
            "type": bool,
            "default": True,
            "description": "Include the quality control summary table.",
        },
        "show_qc_plots": {
            "type": bool,
            "default": True,
            "description": "Include the QC histogram plots section.",
        },
        "show_cross_section_plots": {
            "type": bool,
            "default": True,
            "description": (
                "Include the cross-section plots section (a full-page A4 figure of "
                "PRES vs TIME panels, each coloured by a variable)."
            ),
        },
        "show_logs": {
            "type": bool,
            "default": True,
            "description": "Include the logfile section.",
        },
        "show_index": {
            "type": bool,
            "default": True,
            "description": (
                "Include the closing index: pelagos-py credit, QC flag glossary, "
                "variable index and glider information."
            ),
        },
    }

    def run(self) -> xr.DataArray:
        #   Required inputs for all other steps
        odir = self.context["global_parameters"]["out_directory"]
        data = self.context.get("data")
        fname_core = self.context["global_parameters"]["filename_core"]

        #   Handle optional parameters
        fname = self.parameters.get("fname")
        if not fname:
            fname = fname_core + ".pdf"
        if not fname.endswith(".pdf"):
            fname += ".pdf"
        fout = odir + fname

        title = self.parameters.get("title")
        if not title:
            title = f"Data report {fname_core.replace('_', ' ')}"

        #   Only show the dataset ID subtitle when one is actually present.
        has_dataset_id = bool(data.attrs.get("dataset_id"))
        if not has_dataset_id:
            self.log_warn(
                "Dataset ID missing from OG1 file. Reporting with unk platform information."
            )
            data.attrs["dataset_id"] = UNKNOWN_DATASET_ID
        subtitle = f"Dataset ID: {data.attrs['dataset_id']}" if has_dataset_id else None

        #   The pipeline exposes its full configuration (pipeline block + the
        #   ordered list of steps) so we can reproduce it and list the steps.
        run_config = self.context.get("pipeline_config", {})
        step_names = [s["name"] for s in run_config.get("steps", [])]

        #   Figures are written to a working folder. By default that is a
        #   temporary directory that is removed once the PDF is built; when
        #   delete_figures is False they are kept beside the report in a
        #   uniquely named folder so repeat runs don't overwrite each other.
        delete_figures = self.parameters.get("delete_figures", True)
        if delete_figures:
            fig_dir = tempfile.mkdtemp(prefix="pelagos_report_figs_")
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            fig_dir = os.path.join(odir, f"{fname_core}_report_figures_{stamp}")
            os.makedirs(fig_dir, exist_ok=True)
        fig_dir = fig_dir + os.sep  #   figure paths are built by string concatenation

        #   Pipeline name and description (from the pipeline config block) are
        #   shown on the title page.
        glob_params = self.context["global_parameters"]

        #   Title-page glider track map. Best-effort: any failure (no
        #   coordinates, cartopy/Natural Earth unavailable) just omits the map.
        try:
            track_map_path = glider_track_map(data, fig_dir)
        except Exception as exc:  # noqa: BLE001 - the map must never break the report
            self.log_warn(f"Could not build the title-page track map: {exc}")
            track_map_path = None

        try:
            #   Build the PDF
            pdf = ReportPDF(
                title=title,
                subtitle=subtitle,
                steps=step_names,
                pipeline_name=glob_params.get("name"),
                pipeline_description=glob_params.get("description"),
                track_map_path=track_map_path,
            )
            pdf.title_page()

            #   Each section is optional and defaults on. Lead with the
            #   cross-section plots (the headline view of the mission), then the
            #   Format Checker results (whenever that step ran), the configuration,
            #   per-step diagnostics, QC summary, plots and logs. Close with a
            #   pelagos-py credit and an index (QC flag glossary, variable index
            #   and glider information).
            if self.parameters.get("show_cross_section_plots", True):
                self.log("Generating cross-section plots.")
                cross_section_section(pdf, data, outdir=fig_dir)

            if (
                self.parameters.get("show_format_check", True)
                and "Format Checker" in step_names
            ):
                format_checker_section(
                    pdf,
                    self.context.get("cc_results"),
                    glob_params.get("cc_file"),
                )

            if self.parameters.get("show_configuration", True):
                config_section(pdf, run_config)

            if self.parameters.get("show_diagnostic_plots", True):
                diagnostics_section(pdf, self.context.get("captured_diagnostics"))

            if self.parameters.get("show_qc_summary", True):
                qc_section(pdf, data)

            if self.parameters.get("show_qc_plots", True):
                self.log("Generating images.")
                make_plots(pdf, data, outdir=fig_dir)

            if self.parameters.get("show_logs", True):
                log_path = odir + self.context["global_parameters"]["log_file"]
                add_log(log_path, pdf)

            if self.parameters.get("show_index", True):
                index_section(pdf, data)

            pdf.output(fout)
            self.log(f"Report written to {fout}")
        finally:
            if delete_figures:
                shutil.rmtree(fig_dir, ignore_errors=True)
            else:
                self.log(f"Report figures kept in {fig_dir}")

        return self.context
