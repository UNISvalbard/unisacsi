from re import Match
from typing import Any, Iterator
import pandas as pd
import xarray as xr
import re
import sys

############################################################################
# NAMING
############################################################################

# dictionary to rename variables
# "variable_name":["list of possible names"]
rename_dict: dict[str, list[str]] = {
    "RECORD": ["Record"],
    "T": ["temperature", "Temperature", "temp", "Temp"],
    "RH": ["relative_humidity", "rel_humidity", "Relative humidity"],
    "Speed": ["speed"],
    "Dir": ["direction", "Direction", "dir"],
    "p": ["pressure", "Pressure"],
    "SW": ["Shortwave"],
    "LW": ["Longwave"],
    "lat": ["Latitude"],
    "lon": ["Longitude"],
    "z": ["Altitude"],
    "OX": ["O2Concentration"],
    "S": ["Salinity", "salinity"],
    "SIGTH": ["density_anomaly"],
    "C": ["conductivity"],
}

unit_rename_dict: dict[str, list[str]] = {
    "Pa": ["Pascal"],
    "deg": ["°", "ø", "degrees"],
    "C": [" Celsius"],
    "V": ["Volts"],
    "%": ["percent"],
    "": ["pratical salinity unit"],
}

# dictinary variables and units {'variable_name': 'unit'}
unit_dict: dict[str, str] = {
    "T": "degC",
    "RH": "%RH",
    "Speed": "m/s",
    "Dir": "deg",
    "p": "hPa",
    "SW": "W/m^2",
    "LW": "W/m^2",
    "z": "m",
    "conductivity": "S/m",
    "SIGTH": "kg/m^3",
    "C": "S/m",
}
generall_variables: list[str] = [
    "T",
    "RH",
    "Humidity",
    "Speed",
    "Dir",
    "p",
    "LW",
    "SW",
    "precip",
    "z",
    "u",
    "v",
    "Heading",
    "SIGTH",
    "OX",
    "S",
]

var_attr: list[str] = [
    "max",
    "min",
    "std",
    "var",
    "mean",
    "avg",
]


def std_names(
    data: pd.DataFrame | xr.Dataset,
    bonus: bool = False,
    add_units: bool = False,
    module: str = "a",
) -> pd.DataFrame | xr.Dataset:
    """Used to standardize the names of the variables of reseved data.

    Args:
        data (pd.DataFrame | xr.Dataset): Data where the variables should be standardized.
        bonus (bool, optional): Adds the serial number. Defaults to False.
            - Used for the HOBO data.
        add_units (bool, optional): If True, adds units to the standardized names. Defaults to False.
        module (str, optional): Module name. Defaults to "a".
            - a for atmospheric data.
            - o for ocean data.

    Returns:
        pd.DataFrame | xr.Dataset:
    """
    # format Var_varAttr [unit]

    flat_rename_dict: dict[str, str] = {
        old: new for new, old_list in rename_dict.items() for old in old_list
    }

    flat_unit_rename_dict: dict[str, str] = {
        old: new for new, old_list in unit_rename_dict.items() for old in old_list
    }

    def _std_name(name: str) -> str:
        """Applies the renaming dictionary to the string.

        Args:
            name (str): String to be renamed.

        Returns:
            str: String with applied naming convention.
        """
        for old, new in flat_rename_dict.items():
            name = name.replace(old, new)

        name = name.strip().replace(" ", "_")
        split_name: list[str] = name.split("_")
        if len(split_name) == 1:
            if "SW" in name:
                split_name = ["SW"] + name.replace("SW", " ").strip().split(" ")
            elif "LW" in name:
                split_name = ["LW"] + name.replace("LW", " ").strip().split(" ")
        existing_variable: list[str] = [
            x for x in generall_variables if x in split_name
        ]
        if len(existing_variable) == 1:
            filterd_name: list[str] = [
                x.lower()
                for x in split_name
                if x not in existing_variable and x not in var_attr
            ]
            var_attr_name: list[str] = [x for x in split_name if x in var_attr]
            split_name = existing_variable + filterd_name + var_attr_name
        elif existing_variable == ["T", "Humidity"]:
            existing_variable = ["T"]
            filterd_name: list[str] = [
                x.lower()
                for x in split_name
                if x not in existing_variable and x not in var_attr
            ]
            var_attr_name: list[str] = [x for x in split_name if x in var_attr]
            split_name = existing_variable + filterd_name + var_attr_name

        name = "_".join(split_name)

        return name

    def _std_power(unit: str) -> str:
        """Translates units wirten with +1 or -1 as a power to ^1 or /^1.

        Args:
            unit (str): String to check and translate.

        Returns:
            str: Standardized unit string.
        """
        plus_minus_index: Iterator[Match[str]] = re.finditer(r"[+-]\d+", unit)
        found: bool = False
        # incase there are spaces in the unit and +1 or -1 there, replace them with +1 (found in IWIN)
        for occurrence in plus_minus_index:
            found = True
            unit = unit.replace(" ", "+1")
            plus_minus_index = re.finditer(r"[+-]\d+", unit)
        edit_unit: str = ""
        below: bool = False  # to check if we're already in the denominator
        last_pos_old_str: int = 0
        for occurrence in plus_minus_index:
            # if there are 2 unitnames without a power after each other a ^1 is added
            # for readability
            if occurrence.group()[0] == "+":
                # in case somebody writes a positiv power after a negativ:
                if below:
                    temp_str: str = ""
                    if not edit_unit[backslash_pos - 1].isdigit():
                        temp_str += "^1"
                    if occurrence.group()[1:] == "1":
                        temp_str += unit[last_pos_old_str : occurrence.start()]
                    else:
                        temp_str += (
                            unit[last_pos_old_str : occurrence.start()]
                            + "^"
                            + occurrence.group()[1:]
                        )
                    if backslash_pos == 1 and edit_unit[0] == "1":
                        edit_unit = temp_str + edit_unit[backslash_pos:]
                    else:
                        edit_unit = (
                            edit_unit[:backslash_pos]
                            + temp_str
                            + edit_unit[backslash_pos:]
                        )
                else:
                    if last_pos_old_str != 0 and not edit_unit[-1].isdigit():
                        edit_unit += "^1"
                    if occurrence.group()[1:] == "1":
                        edit_unit += unit[last_pos_old_str : occurrence.start()]
                    else:
                        edit_unit += (
                            unit[last_pos_old_str : occurrence.start()]
                            + "^"
                            + occurrence.group()[1:]
                        )
            elif occurrence.group()[0] == "-":
                # if there are 2 unitnames without a power after each other a ^1 is added
                # for readability
                if below and not edit_unit[-1].isdigit():
                    edit_unit += "^1"
                if last_pos_old_str == 0:
                    edit_unit += "1/"
                    below = True
                    backslash_pos: int = 1
                elif not below and last_pos_old_str != 0:
                    edit_unit += "/"
                    below = True
                    backslash_pos: int = len(edit_unit) - 1
                if occurrence.group()[1:] == "1":
                    edit_unit += unit[last_pos_old_str : occurrence.start()]
                else:
                    edit_unit += (
                        unit[last_pos_old_str : occurrence.start()]
                        + "^"
                        + occurrence.group()[1:]
                    )
            edit_unit = edit_unit.strip()
            last_pos_old_str = occurrence.end()
        if found and last_pos_old_str != len(unit):
            if not edit_unit[-1].isdigit():
                edit_unit += "^1"
            edit_unit += unit[last_pos_old_str:]
        if not found:
            edit_unit = unit

        return edit_unit

    if isinstance(data, pd.DataFrame):
        name_list: list[str] = []
        for oldname in data.columns:
            match_splitname: re.Match[str] | None = None
            # checking for specific patterns
            # for hobo:
            if match_splitname := re.match(r"^(.*?), (.*?) \((.*?)\)$", oldname):
                # [Name, sensor info, unit]
                splitname = [
                    match_splitname.groups()[0].strip(),
                    match_splitname.groups()[2].strip(),
                    match_splitname.groups()[1],
                ]
            # for cambell, eddypro_full_output:
            elif match_splitname := re.match(r"^(.*?)\[(.*?)\](.*?)$", oldname):
                # [Name, unit]
                splitname = [
                    (
                        match_splitname.groups()[0].strip()
                        + " "
                        + match_splitname.groups()[2].strip()
                    ).strip(),
                    match_splitname.groups()[1],
                ]
            # for radiosone
            elif match_splitname := re.match(r"^(.*?)\((.*?)\)(.*?)$", oldname):
                # [Name, unit]
                splitname: list[str] = [
                    (
                        match_splitname.groups()[0].strip()
                        + " "
                        + match_splitname.groups()[2].strip()
                    ).strip(),
                    match_splitname.groups()[1],
                ]
            else:
                splitname = [oldname]

            # processing name
            splitname[0] = _std_name(splitname[0])

            if len(splitname) == 1:
                if splitname == ["#"]:
                    splitname = ["RECORD"]

            elif len(splitname) >= 2:
                # processing units
                splitname[-1] = f"[{_std_power(splitname[-1])}]"
                for old, new in flat_unit_rename_dict.items():
                    splitname[-1] = splitname[-1].replace(old, new)
                # to get the uniformness for HOBO data
                if "RH" in splitname[0] and splitname[-1] == "[%]":
                    splitname[-1] = "[%RH]"
                elif splitname[-1] == "[C]" and any(
                    [x == "T" for x in splitname[0].split("_")]
                ):
                    splitname[-1] = "[degC]"
            if len(splitname) == 3:
                # dealing with sn
                if bonus:
                    sn_num: str = splitname.pop(1).split(" ")[5]
                    splitname[0] += f"_sn{sn_num}"
                else:
                    splitname.pop(1)

            if add_units and not re.search(r"\[.*?\]", splitname[-1]):
                split_var_name: list[str] = splitname[0].split("_")
                if module == "o":
                    unit_dict["p"] = "dbar"
                for var, unit in unit_dict.items():
                    if var in split_var_name:
                        splitname.append(f"[{unit}]")
                        break
                if (
                    not re.search(r"\[.*?\]", splitname[-1])
                    and not splitname[0] == "RECORD"
                ):
                    splitname.append(f"[]")

            name_list.append(" ".join(splitname))
        data.columns = name_list
    elif isinstance(data, xr.Dataset):
        # {old_name:new_name}
        ds_rename_dict: dict[str, str] = {}
        for oldname in data.data_vars:
            ds_rename_dict[oldname] = _std_name(oldname)
            unit: str | None = data[oldname].attrs.get("units", None)
            if unit:
                for old, new in flat_unit_rename_dict.items():
                    unit = unit.replace(old, new)
                # to get the uniformness for HOBO data
                if "RH" in ds_rename_dict[oldname] and unit == "%":
                    unit = "%RH"
                data[oldname].attrs["units"] = _std_power(unit)

        data = data.rename(ds_rename_dict)

    else:
        raise TypeError(f"This functioin is not applicable to {type(data).__name__}.")

    return data


def progress_bar(iteration: float, total: float, length: float = 40) -> None:
    """Print a progress bar.

    Args:
        iteration (float): Current iteration.
        total (_type_): Total iterations.
        length (int, optional): Lenght of bar. Defaults to 40.
    """
    if not isinstance(iteration, (float, int)):
        raise ValueError(
            f"Iteration must be a float or int, not {type(iteration).__name__}."
        )
    if iteration < 0:
        raise ValueError(f"Iteration must be greater than 0, not {iteration}.")
    if iteration > total:
        raise ValueError(f"Iteration must be less than total, not {iteration}.")
    if not isinstance(total, (float, int)):
        raise ValueError(f"Total must be a float or int, not {type(total).__name__}.")
    if total < 0:
        raise ValueError(f"Total must be greater than 0, not {total}.")
    if not isinstance(length, (float, int)):
        raise ValueError(f"Length must be a float or int, not {type(length).__name__}.")
    if length < 0:
        raise ValueError(f"Length must be greater than 0, not {length}.")

    percent: str = f"{100 * (iteration / float(total)):.1f}"
    filled: int = int(length * iteration // total)
    bar: str = "█" * filled + "-" * (length - filled)
    print(f"\r|{bar}| {percent}% Complete", end="\r", flush=True)
    if iteration == total:
        print("Done", " " * (length + 13), flush=True)
    return None
