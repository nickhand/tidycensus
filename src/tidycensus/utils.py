from re import match

import pandas as pd
from loguru import logger

from . import DATA_DIR


def verify_list_inputs(param):
    if param is None:
        param = []
    if isinstance(param, str):
        param = [param]
    return param


# Called to check to see if "state" is a FIPS code, full name or abbreviation.
#
# returns NULL if input is NULL
# returns valid state FIPS code if input is even pseud-valid (i.e. single digit but w/in range)
# returns error if input is not a valid FIPS code
def validate_state(state):

    # Load FIPS state table
    FIPS_STATE_TABLE = pd.read_csv(DATA_DIR / "fips_state_table.csv", dtype=str)

    state = state.strip().lower()  # forgive white space

    if match("^\d+$", state):  # we prbly have FIPS

        state = f"{int(state):02d}"  # forgive 1-digit FIPS codes

        if state in FIPS_STATE_TABLE["fips"].values:
            return state
        else:
            # perhaps they passed in a county FIPS by accident so forgive that, too,
            # but warn the caller
            state_sub = state[:2]
            if state_sub in FIPS_STATE_TABLE["fips"].values:
                name = FIPS_STATE_TABLE.query("fips == @state_sub")["name"].squeeze()
                logger.warning(
                    f"Using first two digits of {state} - '{state_sub}' ({name}) - for FIPS code."
                )
                return state_sub
            else:
                raise ValueError(
                    f"'{state}' is not a valid FIPS code or state name/abbreviation"
                )

    elif match("^\w+", state):  # we might have state abbrev or name

        if (
            len(state) == 2 and state in FIPS_STATE_TABLE["abb"].values
        ):  # yay, an abbrev!
            fips = FIPS_STATE_TABLE.query("abb == @state")["fips"].squeeze()
            logger.info(f"Using FIPS code '{fips}' for state '{state.upper()}'")
            return fips

        elif (
            len(state) > 2 and state in FIPS_STATE_TABLE["name"].values
        ):  # yay, a name!

            fips = FIPS_STATE_TABLE.query("name == @state")["fips"].squeeze()
            logger.info(f"Using FIPS code '{fips}' for state '{state.capitalize()}'")
            return fips
        else:
            raise ValueError(
                f"'{state}' is not a valid FIPS code or state name/abbreviation"
            )

    else:
        raise ValueError(
            f"'{state}' is not a valid FIPS code or state name/abbreviation"
        )


# Some work on a validate_county function
#
#
def validate_county(state, county):

    # Get the state of the county
    state = validate_state(state)

    # Load FIPS codes
    FIPS_CODES = pd.read_csv(DATA_DIR / "fips_codes.csv", dtype=str)

    # Get a df for the requested state to work with
    COUNTY_TABLE = FIPS_CODES.query("state_code == @state")

    if match("^\d+$", county):  # probably a FIPS code

        county = f"{county:03d}"  # in case they passed in 1 or 2 digit county codes

        if county in COUNTY_TABLE["county_code"].values:
            return county
        else:
            state = COUNTY_TABLE.iloc[0]["state_name"]
            logger.warning(
                f"'{county}' is not a current FIPS code for counties in {state}"
            )
        return county

    elif match("^\w+", county):  # should be a county name

        county_index = COUNTY_TABLE["county"].str.match(f"^{county}", case=False)
        matching_counties = COUNTY_TABLE.loc[county_index][
            "county"
        ]  # Get the counties that match

        if len(matching_counties) == 0:
            state = COUNTY_TABLE.iloc[0]["state_name"]
            raise ValueError(f"'{county}' is not a valid name for counties in {state}")

        elif len(matching_counties) == 1:

            matched_county = matching_counties.iloc[0]
            fips = COUNTY_TABLE.query(f"county == '{matched_county}'")[
                "county_code"
            ].squeeze()
            logger.info(f"Using FIPS code '{fips}' for '{matched_county}'")
            return fips

        elif len(matching_counties) > 1:
            raise ValueError(
                f"Your county string matches: {matching_counties.tolist()}. Please refine your selection."
            )
