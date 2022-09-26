import os
import json
from pathlib import Path
from typing import Dict
import xlsxwriter
import numpy as np
import pandas as pd
import config


# ----- Export to JSON file
def export_json(filepath, data={}):
    try:
        # create folder if not exists
        fp, _ = os.path.split(filepath)
        Path(fp).mkdir(parents=True, exist_ok=True)
        # dump data to json
        with open(filepath, "w") as outfile:
            json.dump(data, outfile, indent=4)
        return f"[INFO] File has been saved at -> {filepath}"
    except Exception as e:
        print(e)
        raise IOError("[ERROR] Cannot export to json file!")


# ----- Export multiple JSON file
def export_multi_json(filepath=[], datas=[]):
    try:
        for i in range(1, len(datas) + 1):
            # create folder if not exists
            fp, fname = os.path.split(filepath[i])
            Path(fp).mkdir(parents=True, exist_ok=True)
            # dump data-i to json
            with open(filepath[i], "w") as outfile:
                json.dump(datas[i], outfile)
            return f"[INFO] File has been saved into -> {filepath[i]}!"
    except Exception as e:
        print(e)
        raise IOError("Cannot export to json file!")


# ----- Export dataframe to excel multisheets at once
def export_multisheet(filepath, datas=[], sheets=[], index_label="No"):
    # create folder if not exists
    fp, fname = os.path.split(filepath)
    file_location = Path(fp).mkdir(parents=True, exist_ok=True)
    writer_acc = pd.ExcelWriter(filepath, engine="xlsxwriter")
    count = 0
    for df in datas:
        df.to_excel(writer_acc, sheet_name=sheets[count], index_label="No")
        # formatter
        workbook = writer_acc.book
        worksheet = writer_acc.sheets[sheets[count]]
        border_fmt = workbook.add_format(
            {"bottom": 1, "top": 1, "left": 1, "right": 1}
        )
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(0, 0, len(df), len(df.columns)),
            {"type": "no_blanks", "format": border_fmt},
        )
        count += 1
    writer_acc.save()
    return f"[INFO] File has been saved into -> {filepath}!"


# ----- Export multiple dataframe to an excel sheet
def export_multiframes(
    filepath,
    datas=[],
    df_title=[],
    space=3,
    index_label="No"
):
    # create folder if not exists
    fp, fname = os.path.split(filepath)
    file_location = Path(fp).mkdir(parents=True, exist_ok=True)
    writer_acc = pd.ExcelWriter(filepath, engine="xlsxwriter")
    count = 0
    row_count = 1
    for df in datas:
        startrow = 1 if count < 1 else row_count
        df.to_excel(
            writer_acc,
            sheet_name="Sheet1",
            index_label=index_label,
            startrow=startrow,
            startcol=0,
        )
        workbook = writer_acc.book
        worksheet = writer_acc.sheets["Sheet1"]
        merge_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        border_fmt = workbook.add_format({"border": 1})

        cell_title_format = {
            "first_row": 0,
            "first_col": 0,
            "last_row": 0,
            "last_col": len(df.columns),
            "data": df_title[count],
            "cell_format": merge_format
        }

        if count > 0:
            cell_title_format = {
                "first_row": row_count - 1,
                "first_col": 0,
                "last_row": row_count - 1,
                "last_col": len(df.columns),
                "data": df_title[count],
                "cell_format": merge_format
            }

        # worksheet.write_string(row_count - 1, 0, df_title[count])
        worksheet.merge_range(**cell_title_format)
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(startrow, 0, len(df), len(df.columns)),
            {"type": "no_blanks", "format": border_fmt},
        )
        count += 1
        row_count = row_count + len(df.index) + space + 1
    # Save
    writer_acc.save()
    return f"[INFO] File has been saved into -> {filepath}!"


# ----- Export single dataframe to csv or excel
def export_data(df, filepath, export_to="xlsx", index_label="No"):
    fp, fname = os.path.split(filepath)
    Path(fp).mkdir(parents=True, exist_ok=True)
    if export_to == "xlsx":
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
        df.to_excel(writer, index_label=index_label)
        # formatter
        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]
        border_fmt = workbook.add_format(
            {"bottom": 1, "top": 1, "left": 1, "right": 1}
        )
        worksheet.conditional_format(
            xlsxwriter.utility.xl_range(0, 0, len(df), len(df.columns)),
            {"type": "no_blanks", "format": border_fmt},
        )
        writer.save()
        return f"[INFO] File has been saved into -> {filepath}!"
    elif export_to == "csv":
        df.to_csv(filepath, index_label="No", sep=";")
        return f"[INFO] File has been saved into -> {filepath}!"
    else:
        raise ValueError("Must be one of xlsx or csv!")


# ----- Advanced export dataframe to csv or excel with options
def export_advanced(
    data: Dict = dict(),
    expand_col: bool = True,
    expand_colname: str = "label",
    location: str = "./prep",
    export_to: str = "csv",
    convert_labels: str = None,
    sep: str = ";",
) -> dict:
    """
    A helper to export data from nested dictionary
    to a csv or excel using pandas, and yes its a multiple
    file exported based on nested dicts on "data" arguments.

    Keyword Arguments:
    - data {Dict(str: dict())} -- dict structure must be like:
            {
                "methodName": {
                    "datasetName": {
                        "colA": list(),
                        "colB": list(),
                        "colC": list(),
                    }
                }
            } (default: {dict()})
    - expand_col {bool} -> (default: {True}):
      Expand a list of list to an individual column
    - expand_colname {str} -> (default: {features}):
      Expand column name must be set if expand_col is True
    - location {str} -> (default: {"./preprocessed/test/"}):
      file path location for exported data
    - export_to {str}:
      file types, must be one of csv or excel (default: {"csv"})
    - sep {str}:
      a separator to csv or excel file (default: {";"})

    Raises:
    - ValueError:
        - data is not a dictionary
        - expand_colname is not set
        - export_to is not one of csv or excel
    """
    if not isinstance(data, dict):
        raise ValueError("data format must be a dictionary!")
    if expand_col and not expand_colname:
        raise ValueError("expand_colname must be set if expand_col is True")
    if export_to not in ("csv", "excel"):
        raise ValueError("export_to must be one of csv or excel")
    if sep not in (";", ","):
        raise ValueError("separator must be delimeter (;) or commas (,)")
    # exported data attrs for analysis
    exported_attrs = {dt: {} for dt in data.keys()}
    # Start exporting data
    for dataset, values in data.items():
        print(f"Exported {dataset} dataset...")
        # convert to dictionary
        df = pd.DataFrame.from_dict(values)
        # index start from 1 :)
        df.index = np.arange(1, len(df) + 1)
        # If expand features to columns is True
        if expand_col:
            df_expand = df[expand_colname].apply(pd.Series)
            df_expand = df_expand.rename(columns=lambda x: "x" + str(x + 1))
            df = pd.concat([df[:], df_expand[:]], axis=1)
            df = df.drop([expand_colname], axis=1)
        if convert_labels:
            df["labels"] = df[convert_labels].replace(config.LABEL_ENCODER)
        # rename each col to a title case
        df.columns = map(str.title, df.columns)
        # create folder if not exists
        file_loc = Path(f"{location}/{dataset}_preprocessed/").mkdir(
            parents=True, exist_ok=True
        )
        try:
            if export_to == "excel":
                file_loc = f"{location}/{dataset}_preprocessed/{dataset}.xlsx"
                df.to_excel(file_loc, encoding="utf-8", index_label="No")
            elif export_to == "csv":
                file_loc = f"{location}/{dataset}_preprocessed/{dataset}.csv"
                df.to_csv(
                    file_loc, encoding="utf-8", index_label="No", sep=sep
                )
            else:
                print("File types must be excel or csv!")
                break
            print(f"[INFO] File has been saved into -> {file_loc}")
        except IOError:
            print("Whooa... error occured!")

        exported_attrs[dataset].update(
            {"location": file_loc, "X": ("X1", f"X{len(df_expand)}")}
        )

    return exported_attrs


# if __name__ == "__main__":
#     test = {
#         "JAFFE": {
#             "features": [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
#             "images": ["a", "b", "c"],
#             "emotions_list": ["Surprised", "Surprised", "Surprised"]
#         },
#         "CK+": {
#             "features": [[11, 22, 33], [11, 22, 33]],
#             "images": ["aa", "bb", "cc"],
#             "emotions_list":  ["Surprised", "Surprised", "Surprised"]
#         }
#     }

#     x = export_data(data=test, expand_col=True)
#     print(x)
