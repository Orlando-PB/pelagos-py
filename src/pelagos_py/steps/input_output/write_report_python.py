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
from fpdf.fonts import FontFace
from datetime import datetime, timezone
import getpass
import platform
import json
from importlib.metadata import version, PackageNotFoundError
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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


def current_info() -> dict:
    """Returns current operator information from when the report is being generated."""

    now = datetime.now(timezone.utc)

    try:
        package_version = version("pelagos_py")
    except PackageNotFoundError:
        package_version = "unknown"

    info = {
        "timestamp_utc": now.isoformat(),
        "user": getpass.getuser(),
        "version": package_version,  #   Normally done with __version__.
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


class ReportPDF(FPDF):
    """An :class:`fpdf.FPDF` subclass with the report's title page and helpers.

    Pages after the title page carry a running header (the report title) and a
    page-number footer. Heading/body/table/image helpers wrap the lower-level
    FPDF primitives so the section builders below stay terse.
    """

    def __init__(self, title="Pipeline Report", subtitle=None, author="Unknown"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.report_title = title
        self.report_subtitle = subtitle
        self.report_author = author
        self.set_auto_page_break(auto=True, margin=15)
        self.set_title(sanitize(title))

    def header(self) -> None:
        """Running header on every page except the title page."""
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, sanitize(self.report_title), align="R")
        self.ln(10)
        self.set_text_color(0)

    def footer(self) -> None:
        """Page-number footer on every page except the title page."""
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)

    def title_page(self) -> None:
        """Write a centred title page (title, subtitle, author, date)."""
        self.add_page()
        self.ln(80)
        self.set_font("Helvetica", "B", 28)
        self.multi_cell(0, 12, sanitize(self.report_title), align="C")
        self.ln(6)
        if self.report_subtitle:
            self.set_font("Helvetica", "", 16)
            self.multi_cell(0, 10, sanitize(self.report_subtitle), align="C")
        self.ln(20)
        self.set_font("Helvetica", "", 12)
        self.multi_cell(0, 8, sanitize(self.report_author), align="C")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.multi_cell(0, 8, stamp, align="C")

    def h2(self, text: str) -> None:
        """Write a level-2 heading."""
        self.ln(2)
        self.set_font("Helvetica", "B", 16)
        self.multi_cell(0, 9, sanitize(text))
        self.ln(2)

    def h3(self, text: str) -> None:
        """Write a level-3 heading."""
        self.ln(1)
        self.set_font("Helvetica", "B", 12)
        self.multi_cell(0, 7, sanitize(text))
        self.ln(1)

    def body(self, text: str) -> None:
        """Write a paragraph of body text."""
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, sanitize(text))
        self.ln(2)

    def code_block(self, text: str) -> None:
        """Write monospaced, verbatim text."""
        self.set_font("Courier", "", 8)
        self.multi_cell(0, 4, sanitize(text))
        self.ln(2)

    def add_table(self, headers, rows, widths=None, font_size=8, align="LEFT") -> None:
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
        """
        self.set_font("Helvetica", "", font_size)
        with self.table(
            col_widths=widths,
            text_align=align,
            headings_style=FontFace(emphasis="BOLD"),
        ) as table:
            table.row([sanitize(h) for h in headers])
            for r in rows:
                table.row([sanitize(c) for c in r])
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


### Section builders


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
    pdf.h2("Quality Control Summary")

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


def run_info_page(pdf: ReportPDF, params_dict: dict, glatters: dict) -> None:
    """
    Write a page dedicated to pipeline run information.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    params_dict : dict
        Dictionary of global pipeline parameters.
    glatters : dict
        Dictionary describing the glider and mission. OG1 includes
        "platform_vocabulary" for consistency.
    """
    pdf.add_page()
    pdf.h2("Pipeline run information")

    run_data = current_info()
    pdf.add_table(
        ["", "Run metadata"],
        [[key, value] for key, value in run_data.items()],
        widths=(30, 70),
    )

    pdf.add_table(
        ["", "Pipeline parameter"],
        [[key, value] for key, value in params_dict.items()],
        widths=(30, 70),
    )

    if "platform_vocabulary" in glatters:  #   May not be in every dataset
        pdf.add_table(
            ["", "Glider information"],
            [[key, value] for key, value in glatters.items()],
            widths=(30, 70),
        )


def add_log(logfile, pdf: ReportPDF, ncols: int = 4) -> None:
    """
    Add and format the logfile as a table.

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
    pdf.h2("Logfile of run")

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

    pdf.add_table(
        ["Time", "Level", "Location", "Message"],
        rows,
        widths=(12, 12, 28, 58),
        font_size=7,
    )


def add_cc(ccfile, pdf: ReportPDF) -> None:
    """
    Add the text of the compliance checker step from 'Format Checker'.

    Parameters
    ----------
    ccfile : str
        Path to the compliance checker output (JSON or plain text).
    pdf : ReportPDF
        The active PDF document being written to.
    """
    pdf.add_page()
    pdf.h2("Compliance Checker results")

    if str(ccfile).endswith(".json"):
        with open(ccfile, mode="r") as f:
            cc_data = json.load(f)

        for cname, test_data in cc_data.items():
            scored = test_data.get("scored_points")
            possible = test_data.get("possible_points")
            pdf.h3(f"{cname}: CC score of {scored}/{possible}")

            rows = []
            seen_names = set()
            for entry in test_data.get("all_priorities", []):
                msgs = entry.get("msgs", [])
                if not msgs:
                    continue

                name = entry.get("name", "Unknown")
                for msg in msgs:
                    #   Repeat the name only on its first message
                    display_name = "" if name in seen_names else name
                    seen_names.add(name)
                    rows.append([display_name, msg])

            if rows:
                pdf.add_table(
                    ["Name", f"{cname} message"],
                    rows,
                    widths=(30, 70),
                )
    else:
        with open(ccfile, "r") as f:
            content = f.read()
        pdf.code_block(content)


### Plot builders (save a figure, return its path)


def basic_geo(data, g_extent, outdir, ext=".png") -> str:
    """
    Create a simple geographic plot using the glider LONGITUDE and LATITUDE.

    Returns
    -------
    str
        Path to the saved figure.
    """
    ax0 = plt.axes(projection=ccrs.PlateCarree())
    ax0.set_extent(g_extent, crs=ccrs.PlateCarree())
    ax0.add_feature(cfeature.LAND.with_scale("110m"))
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax0.coastlines(resolution="110m")
    ax0.scatter(
        data["LONGITUDE"],
        data["LATITUDE"],
        s=5,
        color="red",
        marker="+",
        transform=ccrs.PlateCarree(),
    )
    plt.title("Glider Track")
    fname = outdir + f"geographic{ext}"
    plt.savefig(fname)
    plt.close()
    return fname


def inset_geo(
    data,
    outdir: str = "./",
    g_extent: list = [7, 25, 54, 65],
    scale: str = "110m",
    ext: str = ".png",
) -> str:
    """
    Create an inset geographic of two plots for additional positional awareness.

    Unlike basic_geo(), this function will create an inset to make it clearer
    where the glider is operating.

    If the chart looks chunky, consider increasing the resolution in the `scale` arg.

    Parameters
    ----------
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to. Defaults to current directory.
    g_extent : list
        Geographic extent for cartopy geographic plot ([lon1, lon2, lat1, lat2]). Defaults to Baltic Sea.
    scale: str
        Resolution for cartopy to use when adding elements ("10m", "50m", "110m")
    ext : str
        Image filetype extension (.png, .svg, etc.)

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig = plt.figure(figsize=(8, 6))

    lon = data["LONGITUDE"].values
    lat = data["LATITUDE"].values
    lon_min = np.nanmin(lon)
    lon_max = np.nanmax(lon)
    lat_min = np.nanmin(lat)
    lat_max = np.nanmax(lat)

    #   Get the middle of the glider track
    lon_mid = 0.5 * (lon_min + lon_max)
    lat_mid = 0.5 * (lat_min + lat_max)
    pad = 0.1  #   Add some padding in degrees
    lon_span = (
        lon_max - lon_min
    ) + 2 * pad  # Full lat/lon sizes spanning the mission range
    lat_span = (lat_max - lat_min) + 2 * pad

    #   Use the larger span to force glider image to be a square
    span = max(lon_span, lat_span)
    track_extent = [
        lon_mid - span / 2,
        lon_mid + span / 2,
        lat_mid - span / 2,
        lat_mid + span / 2,
    ]  #   Glider data track extent

    #   Glider track on main axes
    ax_main = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_main.set_extent(track_extent, crs=ccrs.PlateCarree())

    ax_main.scatter(
        lon,
        lat,
        s=5,
        color="red",
        marker="+",
        transform=ccrs.PlateCarree(),
    )
    gl = ax_main.gridlines(
        draw_labels=True,  # show tick labels
        dms=True,  # degrees, minutes, seconds
        x_inline=False,
        y_inline=False,
        linewidth=0.5,
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True

    ax_main.coastlines(resolution=scale)
    ax_main.set_title(f"Glider Track: {data.attrs['dataset_id']}")

    # Inset figure on new axes
    inset_ax = fig.add_axes(
        [0.23, 0.05, 0.2, 0.3],  # HA low, VA low, HA hi, VA hi
        projection=ccrs.PlateCarree(),
    )

    inset_ax.set_extent(g_extent, crs=ccrs.PlateCarree())  #   Default Batlic sea
    inset_ax.add_feature(cfeature.BORDERS.with_scale(scale), linewidth=0.5)
    inset_ax.add_feature(cfeature.LAND.with_scale(scale))
    inset_ax.add_feature(cfeature.LAKES.with_scale(scale))
    inset_ax.coastlines(resolution=scale)

    inset_ax.plot(
        [
            track_extent[0],
            track_extent[1],
            track_extent[1],
            track_extent[0],
            track_extent[0],
        ],
        [
            track_extent[2],
            track_extent[2],
            track_extent[3],
            track_extent[3],
            track_extent[2],
        ],
        transform=ccrs.PlateCarree(),
        color="red",
        linewidth=1.2,
    )  # Draw box on top of inset

    #   Save the figure and return the path
    fname = outdir + f"geographic{ext}"
    plt.savefig(fname)
    plt.close(fig)

    return fname


def qc_hist(
    data: xr.Dataset,
    outdir: str,
    var: str,
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

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), layout="constrained")

    #   Prepare the histogram
    ylims = [1, len(data[var])]  #   Log axis cannot be 0
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
    axs[0].set_title(f"{var_source}: n={len(data[var_source])}", ha="right")

    if np.all(np.isnan(data[var])):
        axs[1].text(0.2, 0.5, f"Flags ({var}) are NaN", transform=axs[1].transAxes)
    else:
        data[var].plot.hist(
            yscale="log", bins=bins, xticks=hislim, xlim=xlims, ylim=ylims, ax=axs[1]
        )
        bars = axs[1].containers[0]  #   Number of points in each bin
        axs[1].bar_label(bars, fontsize=7, label_type="center")
        axs[1].set_yscale("log")
    axs[1].set_title(f"{var} flag histogram", ha="right")
    fig.supylabel(data.attrs["dataset_id"])

    fname = outdir + var + ext
    plt.savefig(fname)  #   Save to the outdir
    plt.close(fig)
    return fname


def make_plots(
    pdf: ReportPDF,
    data: xr.Dataset,
    outdir: str,
    extent: list = [7, 25, 54, 65],
) -> None:
    """
    Wrapper for plotting glider QC variables quickly.

    There are millions of points per variable, which xarray can plot very quickly
    in specific ways. Here, geographic and QC histograms are explored.

    Parameters
    ----------
    pdf : ReportPDF
        The active PDF document being written to.
    data : xarray.core.dataset.Dataset
        The entire dataset, including attributes
    outdir : str
        The path to return figures to
    extent : list
        Geographic extent for cartopy geographic plot. Defaults to Baltic Sea.

    TODO: Define long-term storage for this. Is `diagnostics` the right place?
    """
    pdf.add_page()
    pdf.h2("Plots")

    geo_img = inset_geo(data, outdir, extent, scale="50m")
    pdf.image_full(geo_img, aspect=6 / 8)

    qc_vars = [var for var in data.data_vars if "_QC" in var]
    for var in tqdm(
        qc_vars,
        colour="green",
        desc=f"\033[97mProgress \033[0m",
        unit="vars",
    ):
        # Any form of scatter takes ~30 sec, stick with xarray.plot for now (no colorbars, alternative color schemes)
        hist_img = qc_hist(data, outdir, var)
        pdf.image_full(hist_img, aspect=4 / 8)


@register_step
class WriteDataReportPython(BaseStep):
    """
    Writes a PDF report summarizing the generic plots and statistics of the data.

    Built directly with fpdf2 (no LaTeX/Sphinx toolchain required).

    Base template:
    * Title page
    * Quality control summary
    * Basic plots
    * Run metadata and pipeline parameters
    * Logfile
    * Compliance checker results (when available)

    Parameters
    ----------
    title: str
        Name of the report (on title page and header).
    fname: str
        Output .pdf filename; defaults to the filename core when blank.
    extent: list
        Geographic [min_lon, max_lon, min_lat, max_lat] for the inset map.
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
        "extent": {
            "type": list,
            "default": None,
            "description": "Geographic [min_lon, max_lon, min_lat, max_lat] for the inset map.",
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

        if "dataset_id" not in data.attrs:
            self.log_warn(
                "Dataset ID missing from OG1 file. Reporting with unk platform information."
            )
            data.attrs["dataset_id"] = "unknown dataset ID"

        extent = self.parameters.get("extent") or [7, 25, 54, 65]

        #   Build the PDF
        pdf = ReportPDF(
            title=title,
            subtitle=f"Dataset ID: {data.attrs.get('dataset_id')}",
            author=current_info().get("user"),
        )
        pdf.title_page()

        pdf.add_page()
        qc_section(pdf, data)

        self.log("Generating images.")
        make_plots(pdf, data, outdir=odir, extent=extent)

        run_info_page(pdf, self.context["global_parameters"], data.attrs)

        log_path = odir + self.context["global_parameters"]["log_file"]
        add_log(log_path, pdf)

        if "cc_file" in self.context.get("global_parameters", {}):
            cc_path = self.context["global_parameters"]["cc_file"]
            add_cc(cc_path, pdf)

        pdf.output(fout)
        self.log(f"Report written to {fout}")

        return self.context
