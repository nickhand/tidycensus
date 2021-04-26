"""Obtain data for the American Community Survey."""
import os
import sys
from itertools import groupby
from re import match

import numpy as np
from loguru import logger
from tryagain import retries

from .loaders import format_variables_acs, load_data_acs
from .utils import verify_list_inputs


# @retries(
#     max_attempts=3,
#     wait=lambda n: 2 ** n,
#     pre_retry_hook=lambda: logger.info("Call failed. Retrying..."),
# )
def get_acs(
    geography,
    variables=None,
    table=None,
    year=2019,
    output="tidy",
    state=None,
    county=None,
    zcta=None,
    place=None,
    cbsa=None,
    key=None,
    moe_level=90,
    survey="acs5",
    show_call=False,
    verbose=False,
):
    """"""
    # Set the logging level to warnings or higher
    if not verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Handle dict variables
    renamed_variables = None
    if isinstance(variables, dict):
        renamed_variables = [k for k in variables]
        variables = [variables[k] for k in renamed_variables]

    # Handle default parameters
    variables, state, county, zcta, place, cbsa = map(
        verify_list_inputs, [variables, state, county, zcta, place, cbsa]
    )

    # Check year values for specific surveys
    if survey == "acs5" and year < 2009:
        raise ValueError(
            "5-year ACS support in tidycensus begins with the 2005-2009 5-year ACS."
        )
    if survey == "acs1":
        if year < 2005:
            raise ValueError(
                "1-year ACS support in tidycensus begins with the 2005 1-year ACS"
            )
        logger.info(
            "The 1-year ACS provides data for geographies with populations of 65,000 and greater."
        )

    # Check for supplemental estimates
    check_vars = any([match("^K\d.", var) for var in variables])
    check_table = table is not None and match("^K\d.", table)
    if check_vars or check_table:
        logger.info(
            (
                "Getting data from the ACS 1-year Supplemental Estimates. "
                "Data are available for geographies with populations of 20,000 and greater."
            )
        )
        survey = "acsse"

    # Log the survey we are using
    if survey == "acs1":
        logger.info(f"Getting data from the {year} 1-year ACS")
    elif survey == "acs5":
        logger.info(f"Getting data from the {year-4}-{year} 5-year ACS")

    # Check for a Census key
    if key is None:
        key = os.getenv("CENSUS_API_KEY")
        if key is None:
            raise ValueError(
                (
                    "A Census API key is required. Obtain one at http://api.census.gov/data/key_signup.html, "
                    "and then supply the key to the `census_api_key` function to use it throughout your tidycensus session."
                )
            )

    # Check inputs for table/variables
    if not len(variables) and table is None:
        raise ValueError(
            "Either a vector of variables or an ACS table must be specified."
        )
    if len(variables) and table is not None:
        raise ValueError(
            "Specify variables or a table to retrieve; they cannot be combined."
        )

    # Rename geographies
    if geography == "cbsa":
        geography = "metropolitan statistical area/micropolitan statistical area"
    if geography == "cbg":
        geography = "block group"
    if geography == "zcta":
        geography = "zip code tabulation area"
    if geography == "puma":
        geography = "public use microdata area"

    # Check inputs for ZTCA
    if geography == "zip code tabulation area" and len(county):
        raise ValueError("ZCTAs are available by state, but not by county.")
    if len(zcta) and geography != "zip code tabulation area":
        raise ValueError(
            "ZCTAs can only be specified when requesting data at the zip code tabulation area-level."
        )

    # if variables from more than one type of table (e.g. "S1701_C03_002" and "B05002_013"))
    # are requested - take care of this under the hood by having the function
    # call itself for "B" variables, "S" variabls and "DP" variables then combining the results
    if len(variables):
        table_identifiers = list(set([var[0] for var in variables]))
        if len(table_identifiers) > 1 and not all(
            s in ["B", "C"] for s in table_identifiers
        ):

            # Check for supplemental estimates
            if any([match("^K\d.", var) for var in variables]):
                raise ValueError(
                    "At the moment, supplemental estimates variables cannot be combined with variables from other datasets."
                )

            logger.info(
                'Fetching data by table type ("B/C", "S", "DP") and combining the result.'
            )

            # split variables by type into list
            patterns = ["^B|^C", "^S", "^D"]
            vars_by_type = list(
                map(
                    lambda pattern: [
                        list(v)
                        for k, v in groupby(
                            variables, key=lambda s: match(pattern, s) is not None
                        )
                        if k
                    ][0],
                    patterns,
                )
            )

            # Get all of the results and combine
            result = pd.concat(
                map(
                    vars_by_type,
                    lambda vars: get_acs(
                        geography,
                        variables=vars,
                        table=table,
                        year=year,
                        output=output,
                        state=state,
                        county=county,
                        zcta=zcta,
                        place=place,
                        cbsa=cbsa,
                        key=key,
                        moe_level=moe_level,
                        survey=survey,
                        show_call=show_call,
                    ),
                )
            )

            # sort so all vars for each GEOID is together
            return result.sort_values("GEOID")

    # If more than one state specified for tracts/block groups take care of
    # this under the hood by having the function
    # call itself and return the result
    if (geography == "tract" or geography == "block group") and len(state) > 1:

        logger.info(f"Fetching {geography} data by state and combining the result.")
        return pd.concat(
            map(
                state,
                lambda s: get_acs(
                    geography,
                    variables=vars,
                    table=table,
                    year=year,
                    output=output,
                    state=s,
                    county=county,
                    zcta=zcta,
                    place=place,
                    cbsa=cbsa,
                    key=key,
                    moe_level=moe_level,
                    survey=survey,
                    show_call=show_call,
                ),
            )
        )

    # This should be cleaned up and combined with some of the code earlier up
    # but we still need to iterate through counties for block groups earlier than 2013
    if year < 2013 and (geography == "block group" and len(county) > 1):

        logger.info("Fetching block group data by county and combining the result.")
        return pd.concat(
            map(
                county,
                lambda c: get_acs(
                    geography,
                    variables=vars,
                    table=table,
                    year=year,
                    output=output,
                    state=state,
                    county=c,
                    zcta=zcta,
                    place=place,
                    cbsa=cbsa,
                    key=key,
                    moe_level=moe_level,
                    survey=survey,
                    show_call=show_call,
                ),
            )
        )

    # Get the margin of error factor
    if moe_level == 90:
        moe_factor = 1
    elif moe_level == 95:
        moe_factor = 1.96 / 1.645
    elif moe_level == 99:
        moe_factor = 2.56 / 1.645
    else:
        raise ValueError(f"`moe_level` must be one of 90, 95, or 99.")

    # Logic for fetching data tables
    if table is not None:
        if match("^S\d.", table):
            survey2 = survey + "/subject"
        elif match("^DP\d.", table):
            survey2 = survey + "/profile"
        elif match("^K\d.", table):
            survey2 = "acsse"
        else:
            survey2 = survey

        # Get the variables
        variables = variables_from_table_acs(table, year, survey2)

    # Handle variable list of any length
    if len(variables) > 24:

        l = []
        start = 0
        while start < len(variables):
            stop = start + 24
            l.append(variables[start:stop])
            start = stop

        # Get the data
        dat = map(
            lambda x: load_data_acs(
                geography,
                format_variables_acs(x),
                key,
                year,
                survey,
                state=state,
                couny=county,
                zcta=zcta,
                place=place,
                cbsa=cbsa,
                show_call=show_call,
            ),
            l,
        )

        result = reduce(
            lambda left, right: pd.merge(
                left, right, on="GEOID", how="outer", suffixes=("", ".y")
            ),
            dat,
        )
    else:

        result = load_data_acs(
            geography,
            format_variables_acs(variables),
            key,
            year,
            survey,
            state=state,
            county=county,
            zcta=zcta,
            place=place,
            cbsa=cbsa,
            show_call=show_call,
        )

    vars2 = format_variables_acs(variables)
    var_vector = vars2.split(",")

    # Format missing
    missing = [
        -111111111,
        -222222222,
        -333333333,
        -444444444,
        -555555555,
        -666666666,
        -777777777,
        -888888888,
        -999999999,
    ]
    for col in var_vector:
        result[col] = result[col].replace(missing, np.nan)

    # Re-order it
    sub = result[["GEOID", "NAME"] + var_vector]

    # Format results
    if output == "tidy":

        result = (
            sub.melt(id_vars=["GEOID", "NAME"], var_name="variable")
            .assign(
                variable2=lambda df: df["variable"].apply(
                    lambda x: "estimate" if x.endswith("E") else "moe"
                ),
                variable=lambda df: df["variable"].str.slice(0, -1),
            )
            .pivot(
                index=["GEOID", "NAME", "variable"], columns="variable2", values="value"
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )

        if "moe" in result.columns:
            result["moe"] *= moe_factor

        if renamed_variables is not None:
            renamed = dict(zip(variables, renamed_variables))
            result["variable"] = result["variable"].replace(renamed)

    elif output == "wide":

        # Remove duplicate columns
        result = sub.loc[:, ~sub.columns.duplicated()]

        # Add as MOE
        moe_vars = [col for col in result if col.endswith("M")]
        result[moe_vars] *= moe_factor

        if renamed_variables is not None:
            for i, variable in enumerate(variables):
                sub = result.filter(regex=f"^{variable}", axis=1)
                new_cols = dict(
                    zip(
                        sub.columns,
                        [
                            col.replace(variable, renamed_variables[i])
                            for col in sub.columns
                        ],
                    )
                )
                result = result.rename(columns=new_cols)

    return result
