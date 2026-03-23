from toolbox.steps.base_step import BaseStep, register_step
from toolbox.utils.qc_handling import QCHandlingMixin
import toolbox.utils.diagnostics as diag

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import tkinter as tk
from scipy.signal import savgol_filter

FIXED_SAVGOL_WINDOW_VERT = 5
FIXED_SAVGOL_WINDOW_HORIZ = 3
FIXED_SAVGOL_POLY = 2
FIXED_MIN_VALID_DEPTH = -0.5
FIXED_MIN_POINTS_VERT = 5
FIXED_MIN_POINTS_HORIZ = 20

COLOUR_UP = "tab:blue"
COLOUR_DOWN = "tab:green"
COLOUR_HORIZONTAL = "tab:purple"
COLOUR_TURNING = "tab:orange"
COLOUR_VELOCITY = "tab:red"
COLOUR_RAW = "tab:gray"
COLOUR_SMOOTH = "black"
MARKER_SIZE = 2
LINE_WIDTH = 1.5

def _parse_windows(win_sizes, cadence):
    cadence_sec = pd.Timedelta(cadence).total_seconds()
    parsed = []
    for w in win_sizes:
        if isinstance(w, str):
            try:
                w_sec = pd.Timedelta(w).total_seconds()
                parsed.append(max(1, int(round(w_sec / cadence_sec))))
            except ValueError:
                parsed.append(int(w))
        else:
            parsed.append(int(w))
    return parsed

def find_profiles_beta(df_sorted, cadence, filter_win_sizes, gradient_thresholds, horiz_grad_thresh, edge_squeeze, dive_scale, max_depth_gap, min_horizontal_duration, min_horizontal_depth, depth_col):
    df = df_sorted[depth_col].resample(cadence).mean().to_frame()
    df[depth_col] = df[depth_col].interpolate(method='linear')

    windows = _parse_windows(filter_win_sizes, cadence)
    med_win, mean_win = windows[0], windows[1]
    
    df["SMOOTH_DEPTH"] = (
        df[depth_col]
        .rolling(window=med_win, center=True).median()
        .rolling(window=mean_win, center=True).mean()
    )

    dt = pd.Timedelta(cadence).total_seconds()
    df["RAW_VEL"] = np.gradient(df["SMOOTH_DEPTH"]) / dt
    df["RAW_VEL"] = df["RAW_VEL"].fillna(0)
    
    df["SMOOTH_VELOCITY"] = savgol_filter(df["RAW_VEL"], FIXED_SAVGOL_WINDOW_VERT, FIXED_SAVGOL_POLY)
    df["SMOOTH_VELOCITY_HORIZ"] = savgol_filter(df["RAW_VEL"], FIXED_SAVGOL_WINDOW_HORIZ, FIXED_SAVGOL_POLY)
    
    vel_crosses_zero = (df["SMOOTH_VELOCITY"] * df["SMOOTH_VELOCITY"].shift(1)) < 0
    pos_grad, neg_grad = gradient_thresholds

    df["STATE"] = "turning"
    df.loc[df["SMOOTH_VELOCITY"] > pos_grad, "STATE"] = "down"
    df.loc[df["SMOOTH_VELOCITY"] < neg_grad, "STATE"] = "up"
    df.loc[(df["SMOOTH_VELOCITY_HORIZ"].abs() <= horiz_grad_thresh) & (df["SMOOTH_DEPTH"] >= min_horizontal_depth), "STATE"] = "horizontal"
    df.loc[(df["SMOOTH_DEPTH"] < FIXED_MIN_VALID_DEPTH) | vel_crosses_zero, "STATE"] = "turning"
    
    df["is_turning"] = (
        ((df["SMOOTH_VELOCITY"] >= neg_grad) & (df["SMOOTH_VELOCITY"] <= pos_grad)) | 
        (df["SMOOTH_DEPTH"] < FIXED_MIN_VALID_DEPTH) |
        vel_crosses_zero
    )

    turn_mask = df["is_turning"].to_numpy(copy=True)
    if edge_squeeze > 0:
        for _ in range(int(edge_squeeze)):
            shifted_left = np.roll(turn_mask, -1)
            shifted_right = np.roll(turn_mask, 1)
            shifted_left[-1] = turn_mask[-1]
            shifted_right[0] = turn_mask[0]
            
            is_edge = turn_mask & (~shifted_left | ~shifted_right)
            is_single = turn_mask & ~shifted_left & ~shifted_right
            
            to_erode = is_edge & ~is_single
            turn_mask[to_erode] = False
            
    df["is_turning"] = turn_mask
    is_profile = ~df["is_turning"]
    profile_starts = is_profile & ~is_profile.shift(1, fill_value=False)
    df["PROFILE_ID"] = profile_starts.cumsum()
    df.loc[df["is_turning"], "PROFILE_ID"] = np.nan

    df_features = df[["PROFILE_ID", "is_turning", "SMOOTH_VELOCITY", "SMOOTH_VELOCITY_HORIZ", "SMOOTH_DEPTH", "STATE"]]
    
    df_out = pd.merge_asof(
        df_sorted, 
        df_features, 
        left_index=True, 
        right_index=True, 
        direction="nearest", 
        tolerance=pd.Timedelta(cadence)
    )

    df_out["VALID_PROFILE"] = np.nan
    df_out["DIRECTION"] = np.nan
    df_out["GRADIENT"] = np.nan
    
    valid_pid_counter = 1
    
    for pid, group in df_out.dropna(subset=["PROFILE_ID"]).groupby("PROFILE_ID"):
        depth_diffs = group[depth_col].diff().abs()
        sub_groups = (depth_diffs > max_depth_gap).fillna(False).cumsum()
        
        for sub_id, sub_group in group.groupby(sub_groups):
            depth_span = sub_group[depth_col].max() - sub_group[depth_col].min()
            point_count = len(sub_group)
            
            if depth_span >= dive_scale and point_count >= FIXED_MIN_POINTS_VERT:
                df_out.loc[sub_group.index, "VALID_PROFILE"] = valid_pid_counter
                x = (sub_group.index - sub_group.index[0]).total_seconds().values
                
                if len(x) > 1:
                    m, _ = np.polyfit(x, sub_group[depth_col].values, 1)
                    df_out.loc[sub_group.index, "GRADIENT"] = m
                    df_out.loc[sub_group.index, "DIRECTION"] = 1 if m < 0 else -1
                    
                valid_pid_counter += 1
            else:
                df_out.loc[sub_group.index, "is_turning"] = True

    unassigned_mask = df_out["VALID_PROFILE"].isna()
    df_out["is_horiz_candidate"] = False
    df_out.loc[unassigned_mask, "is_horiz_candidate"] = (
        (df_out.loc[unassigned_mask, "SMOOTH_VELOCITY_HORIZ"].abs() <= horiz_grad_thresh) & 
        (df_out.loc[unassigned_mask, "SMOOTH_DEPTH"] >= min_horizontal_depth)
    )

    horiz_groups = (~df_out["is_horiz_candidate"]).cumsum()
    duration_threshold = pd.Timedelta(min_horizontal_duration)

    for sub_id, sub_group in df_out[df_out["is_horiz_candidate"]].groupby(horiz_groups):
        if len(sub_group) < FIXED_MIN_POINTS_HORIZ:
            continue
            
        time_span = sub_group.index[-1] - sub_group.index[0]
        
        if time_span >= duration_threshold:
            df_out.loc[sub_group.index, "VALID_PROFILE"] = valid_pid_counter
            x = (sub_group.index - sub_group.index[0]).total_seconds().values
            
            if len(x) > 1:
                m, _ = np.polyfit(x, sub_group[depth_col].values, 1)
                df_out.loc[sub_group.index, "GRADIENT"] = m
            else:
                df_out.loc[sub_group.index, "GRADIENT"] = 0.0
                
            df_out.loc[sub_group.index, "DIRECTION"] = 0
            df_out.loc[sub_group.index, "is_turning"] = False
            valid_pid_counter += 1

    valid_mask = df_out["VALID_PROFILE"].notna()
    profile_transitions = valid_mask & (df_out["VALID_PROFILE"] != df_out["VALID_PROFILE"].shift(1))
    
    df_out["CHRONO_ID"] = profile_transitions.cumsum()
    df_out.loc[~valid_mask, "CHRONO_ID"] = np.nan
    
    df_out = df_out.drop(columns=["PROFILE_ID", "is_horiz_candidate", "VALID_PROFILE"])
    df_out = df_out.rename(columns={"CHRONO_ID": "PROFILE_ID"})

    return df_out, df

@register_step
class FindProfilesBetaStep(BaseStep, QCHandlingMixin):
    step_name = "Find Profiles Beta"
    
    parameter_schema = {
        "depth_column": {"type": str, "default": "PRES", "description": "Name of the depth column"},
        "resample_cadence": {"type": str, "default": "30s"},
        "gradient_thresholds": {"type": list, "default": [0.033, -0.033]},
        "horiz_gradient_threshold": {"type": float, "default": 0.01},
        "filter_window_sizes": {"type": list, "default": [1, 2]},
        "edge_squeeze": {"type": int, "default": 0},
        "dive_scale": {"type": float, "default": 15.0},
        "max_depth_gap": {"type": float, "default": 60.0},
        "min_horizontal_duration": {"type": str, "default": "20min"},
        "min_horizontal_depth": {"type": float, "default": 1.0}
    }

    def run(self):
        self.log("Attempting to designate profile numbers, directions, and gradients")
        self.filter_qc()

        if not self.depth_column:
            if "PRES_ENG" in self.data.variables:
                self.depth_column = "PRES_ENG"
            elif "PRES" in self.data.variables:
                self.depth_column = "PRES"
            else:
                raise ValueError("Neither PRES_ENG nor PRES variables found in the dataset.")
        elif self.depth_column not in self.data.variables:
            raise ValueError(f"Specified depth column '{self.depth_column}' not found in the dataset.")

        if self.depth_column == "PRES_ENG" and "PRES" in self.data.variables:
            pres_max = float(self.data["PRES"].max())
            eng_max = float(self.data["PRES_ENG"].max())
            ratio = pres_max / eng_max if eng_max != 0 else 1
            if 8 < ratio < 12:
                self.data["PRES_ENG"] = self.data["PRES_ENG"] * 10

        if self.diagnostics:
            if self.is_web_mode():
                self.web_diagnostic_loop()
            else:
                root = self.launch_interactive_gui()
                root.mainloop()

        df_raw = self.data[["TIME", self.depth_column]].to_dataframe().reset_index()
        df_sorted = df_raw.dropna(subset=[self.depth_column, "TIME"]).sort_values("TIME").set_index("TIME")

        df_out, _ = find_profiles_beta(
            df_sorted, self.resample_cadence, self.filter_window_sizes, 
            self.gradient_thresholds, self.horiz_gradient_threshold, self.edge_squeeze,
            self.dive_scale, self.max_depth_gap, self.min_horizontal_duration,
            self.min_horizontal_depth, self.depth_column
        )

        df_out = df_out.reset_index()
        df_final = df_raw.merge(
            df_out[["N_MEASUREMENTS", "PROFILE_ID", "DIRECTION", "GRADIENT"]], 
            on="N_MEASUREMENTS", 
            how="left"
        )

        self.data["PROFILE_NUMBER"] = (("N_MEASUREMENTS",), df_final["PROFILE_ID"].to_numpy())
        self.data.PROFILE_NUMBER.attrs = {
            "long_name": "Derived profile number. NaN indicates no profile.",
            "units": "None",
            "standard_name": "Profile Number",
            "valid_min": 1,
            "valid_max": np.inf,
        }

        self.data["PROFILE_DIRECTION"] = (("N_MEASUREMENTS",), df_final["DIRECTION"].to_numpy())
        self.data.PROFILE_DIRECTION.attrs = {
            "long_name": "Profile Direction (-1: Descending, 0: Horizontal, 1: Ascending, NaN: Not Profile)",
            "units": "None",
        }

        self.data["PROFILE_GRADIENT"] = (("N_MEASUREMENTS",), df_final["GRADIENT"].to_numpy())
        self.data.PROFILE_GRADIENT.attrs = {
            "long_name": "Profile Vertical Gradient",
            "units": "m/s",
        }

        self.generate_qc({
            "PROFILE_NUMBER_QC": ["TIME_QC", f"{self.depth_column}_QC"],
            "PROFILE_DIRECTION_QC": ["TIME_QC", f"{self.depth_column}_QC"],
            "PROFILE_GRADIENT_QC": ["TIME_QC", f"{self.depth_column}_QC"]
        })

        self.context["data"] = self.data
        return self.context

    def create_diagnostic_plot(self):
        df_raw = self.data[["TIME", self.depth_column]].to_dataframe().reset_index()
        df_sorted = df_raw.dropna(subset=[self.depth_column, "TIME"]).sort_values("TIME").set_index("TIME")

        df_out, df_smooth = find_profiles_beta(
            df_sorted, self.resample_cadence, self.filter_window_sizes, 
            self.gradient_thresholds, self.horiz_gradient_threshold, self.edge_squeeze,
            self.dive_scale, self.max_depth_gap, self.min_horizontal_duration, 
            self.min_horizontal_depth, self.depth_column
        )

        fig_main, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})

        x_num = mdates.date2num(df_smooth.index)
        points = np.array([x_num, -df_smooth["SMOOTH_DEPTH"].values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        c_map = {"up": COLOUR_UP, "down": COLOUR_DOWN, "horizontal": COLOUR_HORIZONTAL, "turning": COLOUR_TURNING}
        colours = [c_map[state] for state in df_smooth["STATE"].iloc[:-1]]
        
        lc = LineCollection(segments, colors=colours, linewidths=LINE_WIDTH, zorder=0, alpha=0.7)
        ax1.add_collection(lc)

        turn_mask = df_out["PROFILE_ID"].isna()
        ax1.plot(df_out[turn_mask].index, -df_out[turn_mask][self.depth_column], marker=".", ls="", ms=MARKER_SIZE, color=COLOUR_RAW, alpha=0.5, zorder=1, label="Unassigned Raw")
        
        for pid in df_out["PROFILE_ID"].dropna().unique():
            mask = df_out["PROFILE_ID"] == pid
            direction = df_out.loc[mask, "DIRECTION"].iloc[0]
            if direction == 1:
                c = COLOUR_UP
            elif direction == -1:
                c = COLOUR_DOWN
            else:
                c = COLOUR_HORIZONTAL
            ax1.plot(df_out[mask].index, -df_out[mask][self.depth_column], marker=".", ls="", ms=MARKER_SIZE+1, color=c, zorder=3)

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color=COLOUR_UP, lw=LINE_WIDTH),
            Line2D([0], [0], color=COLOUR_DOWN, lw=LINE_WIDTH),
            Line2D([0], [0], color=COLOUR_HORIZONTAL, lw=LINE_WIDTH),
            Line2D([0], [0], color=COLOUR_TURNING, lw=LINE_WIDTH),
            Line2D([0], [0], marker='.', color='w', markerfacecolor=COLOUR_RAW, markersize=MARKER_SIZE+5)
        ]
        ax1.legend(custom_lines, ['Intended Ascent', 'Intended Descent', 'Intended Horizontal', 'Intended Turning', 'Unassigned Raw'], loc="upper right")

        ax1.set_ylabel(self.depth_column)
        ax1.set_title("Profile Classification Overlay")

        ax2.plot(df_smooth.index, df_smooth["SMOOTH_VELOCITY"], color=COLOUR_VELOCITY, lw=LINE_WIDTH, label="Smoothed Velocity (Vert)")
        ax2.axhline(self.gradient_thresholds[0], color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
        ax2.axhline(self.gradient_thresholds[1], color=COLOUR_TURNING, lw=0.8, ls="--", alpha=0.5)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("Velocity")
        ax2.legend(loc="upper right")

        ax3.plot(df_out.index, df_out["PROFILE_ID"], color="gray", marker=".", ls="", ms=MARKER_SIZE)
        ax3.set_ylabel("Profile ID")
        ax3.set_xlabel("Time")

        fig_main.tight_layout()
        return fig_main

    def launch_interactive_gui(self):
        mpl.use("TkAgg")

        def update_plot():
            fig = self.create_diagnostic_plot()
            fig.show()

        root = tk.Tk()
        root.title("Parameter Adjustment")
        
        entries = {}

        tk.Label(root, text="Cadence").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ent_cadence = tk.Entry(root, width=8)
        ent_cadence.insert(0, self.resample_cadence)
        ent_cadence.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        entries["resample_cadence"] = ent_cadence

        tk.Label(root, text="Vert Grad +/-").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        ent_grad_pos = tk.Entry(root, width=6)
        ent_grad_pos.insert(0, str(self.gradient_thresholds[0]))
        ent_grad_pos.grid(row=0, column=3, sticky="w", padx=5)
        entries["grad_pos"] = ent_grad_pos
        
        ent_grad_neg = tk.Entry(root, width=6)
        ent_grad_neg.insert(0, str(self.gradient_thresholds[1]))
        ent_grad_neg.grid(row=0, column=4, sticky="w", padx=5)
        entries["grad_neg"] = ent_grad_neg

        tk.Label(root, text="Win Med/Mean").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ent_win_med = tk.Entry(root, width=6)
        ent_win_med.insert(0, str(self.filter_window_sizes[0]))
        ent_win_med.grid(row=1, column=1, sticky="w", padx=5)
        entries["win_med"] = ent_win_med

        ent_win_mean = tk.Entry(root, width=6)
        ent_win_mean.insert(0, str(self.filter_window_sizes[1]))
        ent_win_mean.grid(row=1, column=2, sticky="w", padx=5)
        entries["win_mean"] = ent_win_mean

        tk.Label(root, text="Dive Scale").grid(row=1, column=3, sticky="e", padx=5, pady=2)
        ent_scale = tk.Entry(root, width=6)
        ent_scale.insert(0, str(self.dive_scale))
        ent_scale.grid(row=1, column=4, sticky="w", padx=5, pady=2)
        entries["dive_scale"] = ent_scale

        tk.Label(root, text="Horiz Grad").grid(row=2, column=0, sticky="e", padx=5, pady=2)
        ent_h_grad = tk.Entry(root, width=6)
        ent_h_grad.insert(0, str(self.horiz_gradient_threshold))
        ent_h_grad.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        entries["horiz_gradient_threshold"] = ent_h_grad

        tk.Label(root, text="Horiz Dur.").grid(row=2, column=2, sticky="e", padx=5, pady=2)
        ent_h_dur = tk.Entry(root, width=8)
        ent_h_dur.insert(0, self.min_horizontal_duration)
        ent_h_dur.grid(row=2, column=3, sticky="w", padx=5, pady=2)
        entries["min_horizontal_duration"] = ent_h_dur

        tk.Label(root, text="Edge Squeeze").grid(row=3, column=0, sticky="e", padx=5, pady=2)
        ent_squeeze = tk.Entry(root, width=6)
        ent_squeeze.insert(0, str(self.edge_squeeze))
        ent_squeeze.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        entries["edge_squeeze"] = ent_squeeze

        tk.Label(root, text="Max Depth Gap").grid(row=3, column=2, sticky="e", padx=5, pady=2)
        ent_gap = tk.Entry(root, width=6)
        ent_gap.insert(0, str(self.max_depth_gap))
        ent_gap.grid(row=3, column=3, sticky="w", padx=5, pady=2)
        entries["max_depth_gap"] = ent_gap

        def focus_next(event):
            event.widget.tk_focusNext().focus()
            return "break"

        def focus_prev(event):
            event.widget.tk_focusPrev().focus()
            return "break"

        root.bind("<Down>", focus_next)
        root.bind("<Up>", focus_prev)

        def on_cancel(event=None):
            plt.close('all')
            root.quit()
            root.destroy()

        def on_regenerate(event=None):
            self.resample_cadence = entries["resample_cadence"].get()
            self.gradient_thresholds = [float(entries["grad_pos"].get()), float(entries["grad_neg"].get())]
            self.horiz_gradient_threshold = float(entries["horiz_gradient_threshold"].get())
            
            med_val = entries["win_med"].get()
            mean_val = entries["win_mean"].get()
            self.filter_window_sizes = [
                med_val if not med_val.isdigit() else int(med_val), 
                mean_val if not mean_val.isdigit() else int(mean_val)
            ]
            
            self.edge_squeeze = int(entries["edge_squeeze"].get())
            self.dive_scale = float(entries["dive_scale"].get())
            self.max_depth_gap = float(entries["max_depth_gap"].get())
            self.min_horizontal_duration = entries["min_horizontal_duration"].get()
            
            plt.close('all')
            update_plot()

        def on_save(event=None):
            self.update_parameters(
                resample_cadence=self.resample_cadence,
                gradient_thresholds=self.gradient_thresholds,
                horiz_gradient_threshold=self.horiz_gradient_threshold,
                filter_window_sizes=self.filter_window_sizes,
                edge_squeeze=self.edge_squeeze,
                dive_scale=self.dive_scale,
                max_depth_gap=self.max_depth_gap,
                min_horizontal_duration=self.min_horizontal_duration
            )
            plt.close('all')
            root.quit()
            root.destroy()

        root.bind("<Return>", on_save)
        root.bind("<Escape>", on_cancel)
        root.bind("<Control-s>", on_save)
        root.bind("<Command-s>", on_save)

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=4, column=0, columnspan=5, pady=15)

        tk.Button(btn_frame, text="Regenerate", command=on_regenerate).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save", command=on_save).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", padx=5)

        update_plot()
        return root