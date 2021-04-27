from re import match, sub

import pandas as pd
from loguru import logger
from requests import get

from .utils import validate_county, validate_state, verify_list_inputs


def format_variables_acs(variables):

    # find code in 'data-raw/no_moe_vars.R' to pull these vars from api
    no_moes = [
        "B00001_001",
        "B00002_001",
        "B98001_001",
        "B98001_002",
        "B98002_001",
        "B98002_002",
        "B98002_003",
        "B98011_001",
        "B98012_001",
        "B98012_002",
        "B98012_003",
        "B98013_001",
        "B98013_002",
        "B98013_003",
        "B98013_004",
        "B98013_005",
        "B98013_006",
        "B98013_007",
        "B98014_001",
        "B98021_001",
        "B98021_002",
        "B98021_003",
        "B98021_004",
        "B98021_005",
        "B98021_006",
        "B98021_007",
        "B98021_008",
        "B98021_009",
        "B98021_010",
        "B98022_001",
        "B98022_002",
        "B98022_003",
        "B98022_004",
        "B98022_005",
        "B98022_006",
        "B98022_007",
        "B98022_008",
        "B98022_009",
        "B98022_010",
        "B98031_001",
        "B98032_001",
        "B99011_001",
        "B99011_002",
        "B99011_003",
        "B99012_001",
        "B99012_002",
        "B99012_003",
        "B99021_001",
        "B99021_002",
        "B99021_003",
        "B99031_001",
        "B99031_002",
        "B99031_003",
        "B99051_001",
        "B99051_002",
        "B99051_003",
        "B99051_004",
        "B99051_005",
        "B99051_006",
        "B99051_007",
        "B99052_001",
        "B99052_002",
        "B99052_003",
        "B99052_004",
        "B99052_005",
        "B99052_006",
        "B99052_007",
        "B99052PR_001",
        "B99052PR_002",
        "B99052PR_003",
        "B99052PR_004",
        "B99052PR_005",
        "B99052PR_006",
        "B99052PR_007",
        "B99053_001",
        "B99053_002",
        "B99053_003",
        "B99061_001",
        "B99061_002",
        "B99061_003",
        "B99071_001",
        "B99071_002",
        "B99071_003",
        "B99072_001",
        "B99072_002",
        "B99072_003",
        "B99072_004",
        "B99072_005",
        "B99072_006",
        "B99072_007",
        "B99080_001",
        "B99080_002",
        "B99080_003",
        "B99081_001",
        "B99081_002",
        "B99081_003",
        "B99081_004",
        "B99081_005",
        "B99082_001",
        "B99082_002",
        "B99082_003",
        "B99082_004",
        "B99082_005",
        "B99083_001",
        "B99083_002",
        "B99083_003",
        "B99083_004",
        "B99083_005",
        "B99084_001",
        "B99084_002",
        "B99084_003",
        "B99084_004",
        "B99084_005",
        "B99085_001",
        "B99085_002",
        "B99085_003",
        "B99086_001",
        "B99086_002",
        "B99086_003",
        "B99087_001",
        "B99087_002",
        "B99087_003",
        "B99087_004",
        "B99087_005",
        "B99088_001",
        "B99088_002",
        "B99088_003",
        "B99088_004",
        "B99088_005",
        "B99089_001",
        "B99089_002",
        "B99089_003",
        "B99092_001",
        "B99092_002",
        "B99092_003",
        "B99102_001",
        "B99102_002",
        "B99102_003",
        "B99103_001",
        "B99103_002",
        "B99103_003",
        "B99104_001",
        "B99104_002",
        "B99104_003",
        "B99104_004",
        "B99104_005",
        "B99104_006",
        "B99104_007",
        "B99121_001",
        "B99121_002",
        "B99121_003",
        "B99122_001",
        "B99122_002",
        "B99122_003",
        "B99123_001",
        "B99123_002",
        "B99123_003",
        "B99124_001",
        "B99124_002",
        "B99124_003",
        "B99125_001",
        "B99125_002",
        "B99125_003",
        "B99126_001",
        "B99126_002",
        "B99126_003",
        "B99131_001",
        "B99131_002",
        "B99131_003",
        "B99132_001",
        "B99132_002",
        "B99132_003",
        "B99141_001",
        "B99141_002",
        "B99141_003",
        "B99142_001",
        "B99142_002",
        "B99142_003",
        "B99151_001",
        "B99151_002",
        "B99151_003",
        "B99152_001",
        "B99152_002",
        "B99152_003",
        "B99161_001",
        "B99161_002",
        "B99161_003",
        "B99162_001",
        "B99162_002",
        "B99162_003",
        "B99162_004",
        "B99162_005",
        "B99162_006",
        "B99162_007",
        "B99163_001",
        "B99163_002",
        "B99163_003",
        "B99163_004",
        "B99163_005",
        "B99171_001",
        "B99171_002",
        "B99171_003",
        "B99171_004",
        "B99171_005",
        "B99171_006",
        "B99171_007",
        "B99171_008",
        "B99171_009",
        "B99171_010",
        "B99171_011",
        "B99171_012",
        "B99171_013",
        "B99171_014",
        "B99171_015",
        "B99172_001",
        "B99172_002",
        "B99172_003",
        "B99172_004",
        "B99172_005",
        "B99172_006",
        "B99172_007",
        "B99172_008",
        "B99172_009",
        "B99172_010",
        "B99172_011",
        "B99172_012",
        "B99172_013",
        "B99172_014",
        "B99172_015",
        "B99181_001",
        "B99181_002",
        "B99181_003",
        "B99182_001",
        "B99182_002",
        "B99182_003",
        "B99183_001",
        "B99183_002",
        "B99183_003",
        "B99184_001",
        "B99184_002",
        "B99184_003",
        "B99185_001",
        "B99185_002",
        "B99185_003",
        "B99186_001",
        "B99186_002",
        "B99186_003",
        "B99187_001",
        "B99187_002",
        "B99187_003",
        "B99187_004",
        "B99187_005",
        "B99187_006",
        "B99187_007",
        "B99191_001",
        "B99191_002",
        "B99191_003",
        "B99191_004",
        "B99191_005",
        "B99191_006",
        "B99191_007",
        "B99191_008",
        "B99192_001",
        "B99192_002",
        "B99192_003",
        "B99192_004",
        "B99192_005",
        "B99192_006",
        "B99192_007",
        "B99192_008",
        "B99193_001",
        "B99193_002",
        "B99193_003",
        "B99193_004",
        "B99193_005",
        "B99193_006",
        "B99193_007",
        "B99193_008",
        "B99194_001",
        "B99194_002",
        "B99194_003",
        "B99194_004",
        "B99194_005",
        "B99194_006",
        "B99194_007",
        "B99194_008",
        "B99201_001",
        "B99201_002",
        "B99201_003",
        "B99201_004",
        "B99201_005",
        "B99201_006",
        "B99201_007",
        "B99201_008",
        "B99211_001",
        "B99211_002",
        "B99211_003",
        "B99212_001",
        "B99212_002",
        "B99212_003",
        "B99221_001",
        "B99221_002",
        "B99221_003",
        "B99231_001",
        "B99231_002",
        "B99231_003",
        "B99232_001",
        "B99232_002",
        "B99232_003",
        "B99233_001",
        "B99233_002",
        "B99233_003",
        "B99233_004",
        "B99233_005",
        "B99234_001",
        "B99234_002",
        "B99234_003",
        "B99234_004",
        "B99234_005",
        "B99241_001",
        "B99241_002",
        "B99241_003",
        "B99242_001",
        "B99242_002",
        "B99242_003",
        "B99243_001",
        "B99243_002",
        "B99243_003",
        "B99244_001",
        "B99244_002",
        "B99244_003",
        "B99245_001",
        "B99245_002",
        "B99245_003",
        "B99246_001",
        "B99246_002",
        "B99246_003",
        "B992510_001",
        "B992510_002",
        "B992510_003",
        "B992511_001",
        "B992511_002",
        "B992511_003",
        "B992512_001",
        "B992512_002",
        "B992512_003",
        "B992513_001",
        "B992513_002",
        "B992513_003",
        "B992514_001",
        "B992514_002",
        "B992514_003",
        "B992515_001",
        "B992515_002",
        "B992515_003",
        "B992516_001",
        "B992516_002",
        "B992516_003",
        "B992518_001",
        "B992518_002",
        "B992518_003",
        "B992519_001",
        "B992519_002",
        "B992519_003",
        "B99252_001",
        "B99252_002",
        "B99252_003",
        "B992520_001",
        "B992520_002",
        "B992520_003",
        "B992521_001",
        "B992521_002",
        "B992521_003",
        "B992522_001",
        "B992522_002",
        "B992522_003",
        "B992522_004",
        "B992522_005",
        "B992522_006",
        "B992522_007",
        "B992523_001",
        "B992523_002",
        "B992523_003",
        "B99253_001",
        "B99253_002",
        "B99253_003",
        "B99254_001",
        "B99254_002",
        "B99254_003",
        "B99255_001",
        "B99255_002",
        "B99255_003",
        "B99256_001",
        "B99256_002",
        "B99256_003",
        "B99257_001",
        "B99257_002",
        "B99257_003",
        "B99258_001",
        "B99258_002",
        "B99258_003",
        "B99259_001",
        "B99259_002",
        "B99259_003",
        "B992701_001",
        "B992701_002",
        "B992701_003",
        "B992702_001",
        "B992702_002",
        "B992702_003",
        "B992703_001",
        "B992703_002",
        "B992703_003",
        "B992704_001",
        "B992704_002",
        "B992704_003",
        "B992705_001",
        "B992705_002",
        "B992705_003",
        "B992706_001",
        "B992706_002",
        "B992706_003",
        "B992707_001",
        "B992707_002",
        "B992707_003",
        "B992708_001",
        "B992708_002",
        "B992708_003",
        "B992709_001",
        "B992709_002",
        "B992709_003",
        "B99281_001",
        "B99281_002",
        "B99281_003",
        "B99282_001",
        "B99282_002",
        "B99282_003",
        "B99282_004",
        "B99282_005",
        "B99282_006",
        "B99282_007",
        "B99282_008",
        "B99282_009",
        "B99283_001",
        "B99283_002",
        "B99283_003",
        "B99283_004",
        "B99283_005",
    ]

    # First, remove E or M if user has put it in
    variables1 = [v[:-1] if v[-1] in ["E", "M"] else v for v in variables]

    # Now, make unique
    variables2 = list(set(variables1))

    # Next, separate into vars with and without MOEs
    variables2a = [v for v in variables2 if v not in no_moes]
    variables2_nomoe = [v for v in variables2 if v in no_moes]

    # Now, expand with both E and M if MOE is applicable
    variables3 = [v + suffix for v in variables2a for suffix in ["E", "M"]]
    if len(variables2_nomoe):
        variables3_nomoe = [var + "E" for var in variables2_nomoe]
        variables3 = variables3 + variables3_nomoe

    # Now, put together all these strings if need be
    return ",".join(variables3)


def load_data_acs(
    geography,
    formatted_variables,
    key,
    year,
    survey,
    state=None,
    county=None,
    zcta=None,
    place=None,
    cbsa=None,
    show_call=False,
):

    # Check inputs
    state, county, zcta, place, cbsa = map(
        verify_list_inputs, [state, county, zcta, place, cbsa]
    )

    # Base URL
    base = f"https://api.census.gov/data/{year}/acs/{survey}"

    if any([match("^DP", var) for var in formatted_variables.split(",")]):
        logger.info("Using the ACS Data Profile")
        base += "/profile"

    if any([match("^S\d.", var) for var in formatted_variables.split(",")]):
        logger.info("Using the ACS Subject Tables")
        base += "/subject"

    for_area = geography + ":*"

    # We have cbsa
    if len(cbsa):

        cbsa = ",".join(cbsa)
        for_area = f"{geography}:{cbsa}"

        vars_to_get = formatted_variables + ",NAME"
        call = get(base, params={"get": vars_to_get, "for": for_area, "key": key})

    # We have a state
    elif len(state):

        state = [validate_state(s) for s in state]
        state = ",".join(state)

        if geography == "state":
            for_area = f"state:{state}"

        # We have a county too
        if len(county):

            county = [validate_county(state, c) for c in county]
            county = ",".join(county)

            if geography == "county":
                for_area = f"county:{county}"
                in_area = f"state:{state}"
            else:
                in_area = f"state:{state}+county:{county}"
        elif len(place):

            place = ",".join(place)

            if geography == "place":
                for_area = f"place:{place}"
                in_area = f"state:{state}"
            else:
                in_area = f"state:{state}+place:{place}"
        else:
            if year > 2013 and geography == "block group" and len(county):
                in_area = f"state:{state}&in=county:*"
            else:
                in_area = f"state:{state}"

        # The variables to get
        vars_to_get = formatted_variables + ",NAME"

        if geography == "state" and state is not None:

            call = get(base, params={"get": vars_to_get, "for": for_area, "key": key})
        else:
            call = get(
                base,
                params={"get": vars_to_get, "for": for_area, "in": in_area, "key": key},
            )

    # We have a ZIP code
    elif len(zcta):

        for_area = ",".join(zcta)
        vars_to_get = formatted_variables + ",NAME"

        call = get(
            base,
            params={"get": vars_to_get, "for": f"{geography}:{for_area}", "key": key},
        )

    else:

        vars_to_get = formatted_variables + ",NAME"
        call = get(
            base, params={"get": vars_to_get, "for": f"{geography}:*", "key": key}
        )

    if show_call:
        call_url = sub("&key.*", "", call.url)
        logger.info(f"Census API call: {call_url}")

    # Make sure call status returns 200, else, print the error message for the user.
    if call.status_code != 200:
        msg = call.text

        if match("The requested resource is not available", msg):
            raise ValueError(
                (
                    "One or more of your requested variables is likely not available"
                    " at the requested geography.  Please refine your selection."
                )
            )
        else:
            raise ValueError(
                f"Your API call has errors. The API message returned is {msg}."
            )

    content = call.text
    if match("You included a key with this request", content):
        raise ValueError(
            (
                "You have supplied an invalid or inactive API key. "
                "To obtain a valid API key, visit https://api.census.gov/data/key_signup.html. "
                "To activate your key, be sure to click the link provided to you in the email from the Census Bureau that contained your key."
            )
        )

    # Convert to dataframe
    dat = pd.read_json(content)
    dat.columns = dat.iloc[0]
    dat = dat.iloc[1:].copy()

    # Set a numeric
    var_vector = formatted_variables.split(",")
    dat[var_vector] = dat[var_vector].astype(float)

    # Get the geography ID variables
    v2 = var_vector + ["NAME"]
    id_vars = [col for col in dat.columns if col not in v2]

    # Paste into a GEOID column
    dat["GEOID"] = dat[id_vars].apply(lambda row: "".join(row.astype(str)), axis=1)

    # Now, remove them
    dat = dat.drop(labels=id_vars, axis=1)

    return dat
