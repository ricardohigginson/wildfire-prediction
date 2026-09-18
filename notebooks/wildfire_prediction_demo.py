# ============================================================
# Wildfire Ignition Risk Prediction — Demo
# ============================================================

# This script contains the reusable logic and interfaces for
# the wildfire prediction demo.
#
# It can be used in two ways:
#
# 1. As a standalone interactive terminal application.
# 2. As a Python module imported by the demo notebook.
#
# The model, thresholds, feature definitions, prediction
# function, SHAP explanation, scenarios and Jupyter interface
# are kept in one place so that the notebook does not duplicate
# the prediction logic.


# ============================================================
# Imports
# ============================================================

import json
import joblib
import shap
import pandas as pd
import ipywidgets as widgets

from IPython.display import display, HTML
from pathlib import Path


# ============================================================
# Load model and metadata
# ============================================================

# Get the directory where this Python file is located.
# This makes all local file paths independent of the current
# working directory.

BASE_DIR = Path(__file__).resolve().parent


# Load the final reduced XGBoost model.

model = joblib.load(
    BASE_DIR / "xgb_reduced_final.pkl"
)


# Load feature metadata and classification thresholds.

with open(
    BASE_DIR / "feature_metadata.json",
    "r"
) as f:
    metadata = json.load(f)


# Extract the model feature list.

features = metadata["features"]


# Extract the classification thresholds.

balanced_threshold = metadata["balanced_threshold"]

high_sensitivity_threshold = (
    metadata["high_sensitivity_threshold"]
)


# ============================================================
# Feature definitions
# ============================================================

# Each feature contains:
#
# 1. Human-readable name
# 2. Unit
# 3. Description
# 4. Explanation of how higher values generally relate
#    to wildfire-conducive conditions
# 5. Minimum accepted input value
# 6. Maximum accepted input value

input_definitions = {

    "Precipitation_Average_30_Days": (
        "30-Day Precipitation",
        "mm/day",
        "Average daily rainfall during the previous 30 days.",
        "Higher values generally indicate wetter conditions and tend to decrease wildfire probability.",
        0,
        100
    ),

    "Specific_Humidity_Average_30_Days": (
        "30-Day Specific Humidity",
        "kg water vapour / kg air",
        "Amount of water vapour in the air relative to the total mass of air.",
        "Higher values indicate more moisture in the air and generally reduce fire-conducive dryness.",
        0,
        0.05
    ),

    "Min_Relative_Humidity_Min_30_Days": (
        "30-Day Minimum Relative Humidity",
        "%",
        "Lowest percentage of moisture in the air during the previous 30 days.",
        "Higher values indicate moister conditions and generally reduce wildfire-conducive dryness.",
        0,
        100
    ),

    "Max_Relative_Humidity_Max_7_Days": (
        "7-Day Maximum Relative Humidity",
        "%",
        "Highest percentage of moisture in the air during the previous 7 days.",
        "Higher values indicate more humid conditions and generally reduce wildfire-conducive dryness.",
        0,
        100
    ),

    "Solar_Radiation_Average_7_Days": (
        "7-Day Solar Radiation",
        "W/m²",
        "Median solar energy reaching the Earth's surface during the previous 7 days.",
        "Higher values indicate greater solar energy input and can contribute to drying conditions.",
        0,
        500
    ),

    "Min_Daily_Temperature_Min_14_Days": (
        "14-Day Minimum Temperature",
        "°C",
        "Lowest daily temperature during the previous 14 days.",
        "Higher minimum temperatures generally indicate warmer conditions that can contribute to fire risk.",
        -50,
        50
    ),

    "Max_Daily_Temperature_Max_30_Days": (
        "30-Day Maximum Temperature",
        "°C",
        "Highest daily temperature during the previous 30 days.",
        "Higher values indicate hotter conditions and generally increase wildfire-conducive conditions.",
        -20,
        60
    ),

    "Wind_Speed_Average_30_Days": (
        "30-Day Wind Speed",
        "m/s",
        "Median wind speed during the previous 30 days.",
        "Higher values indicate stronger winds, which can increase fire spread and wildfire risk.",
        0,
        50
    ),

    "Vapor_Pressure_Deficit_Average_30_Days": (
        "30-Day Vapor Pressure Deficit",
        "kPa",
        "Indicates how strongly the air can draw moisture from vegetation.",
        "Higher values indicate drier atmospheric conditions and generally increase wildfire-conducive conditions.",
        0,
        10
    ),

    "Potential_Evapotranspiration_Average_30_Days": (
        "30-Day Potential Evapotranspiration",
        "mm/day",
        "Estimated amount of water that could evaporate from the surface and vegetation.",
        "Higher values indicate greater atmospheric demand for water and can contribute to drying conditions.",
        0,
        20
    ),

    "Solar_Radiation_Today": (
        "Solar Radiation — Today",
        "W/m²",
        "Amount of solar energy reaching the Earth's surface today.",
        "Higher values indicate greater solar energy input and can contribute to drying conditions.",
        0,
        500
    ),

    "Min_Relative_Humidity_Today": (
        "Minimum Relative Humidity — Today",
        "%",
        "Lowest percentage of moisture in the air today.",
        "Higher values indicate moister conditions and generally reduce wildfire-conducive dryness.",
        0,
        100
    ),

    "Vapor_Pressure_Deficit_Today": (
        "Vapor Pressure Deficit — Today",
        "kPa",
        "Indicates how strongly the air can draw moisture from vegetation today.",
        "Higher values indicate drier atmospheric conditions and generally increase wildfire-conducive conditions.",
        0,
        10
    ),

    "Wind_Speed_Today": (
        "Wind Speed — Today",
        "m/s",
        "Wind speed recorded today.",
        "Higher values indicate stronger winds, which can increase fire spread and wildfire risk.",
        0,
        50
    )
}


# ============================================================
# Human-readable feature names
# ============================================================

display_names = {

    "Precipitation_Average_30_Days":
        "30-Day Precipitation",

    "Specific_Humidity_Average_30_Days":
        "30-Day Specific Humidity",

    "Min_Relative_Humidity_Min_30_Days":
        "30-Day Minimum Relative Humidity",

    "Max_Relative_Humidity_Max_7_Days":
        "7-Day Maximum Relative Humidity",

    "Solar_Radiation_Average_7_Days":
        "7-Day Solar Radiation",

    "Min_Daily_Temperature_Min_14_Days":
        "14-Day Minimum Temperature",

    "Max_Daily_Temperature_Max_30_Days":
        "30-Day Maximum Temperature",

    "Wind_Speed_Average_30_Days":
        "30-Day Wind Speed",

    "Vapor_Pressure_Deficit_Average_30_Days":
        "30-Day Vapor Pressure Deficit",

    "Potential_Evapotranspiration_Average_30_Days":
        "30-Day Potential Evapotranspiration",

    "Solar_Radiation_Today":
        "Solar Radiation — Today",

    "Min_Relative_Humidity_Today":
        "Minimum Relative Humidity — Today",

    "Vapor_Pressure_Deficit_Today":
        "Vapor Pressure Deficit — Today",

    "Wind_Speed_Today":
        "Wind Speed — Today"
}


# Create a dictionary containing the unit for each feature.

units = {
    feature: input_definitions[feature][1]
    for feature in features
}


# ============================================================
# SHAP explainer
# ============================================================

# Create the SHAP explainer once when the module is loaded.
# This avoids recreating it for every prediction.

explainer = shap.TreeExplainer(model)


# ============================================================
# Prediction function
# ============================================================

def predict_wildfire(input_data, threshold):
    """
    Generate a wildfire prediction and SHAP explanation.

    Parameters
    ----------
    input_data : dict or pandas.DataFrame
        Environmental conditions for the prediction.

    threshold : float
        Probability threshold used to classify the prediction.

    Returns
    -------
    dict
        Dictionary containing:
        - probability
        - prediction
        - shap_values
        - top_5
        - input_df
    """

    # --------------------------------------------------------
    # Validate and prepare the input
    # --------------------------------------------------------

    if isinstance(input_data, pd.DataFrame):

        input_df = input_data.copy()

    else:

        # Check that every model feature is present.

        missing = set(features) - set(input_data.keys())

        if missing:

            raise ValueError(
                f"Missing features in input: {missing}"
            )

        # Create a one-row DataFrame using the exact
        # feature order expected by the model.

        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )


    # Ensure exact model features and order.

    input_df = input_df[features]


    # --------------------------------------------------------
    # Check for missing values
    # --------------------------------------------------------

    if input_df.isna().any().any():

        missing_columns = (
            input_df.columns[
                input_df.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Missing values detected in input features: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # Generate wildfire probability
    # --------------------------------------------------------

    # Class 0 = No Wildfire
    # Class 1 = Wildfire

    wildfire_probability = model.predict_proba(
        input_df
    )[0, 1]


    # --------------------------------------------------------
    # Apply classification threshold
    # --------------------------------------------------------

    prediction = int(
        wildfire_probability >= threshold
    )


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    shap_values = explainer.shap_values(
        input_df
    )


    # Create a table containing the SHAP value
    # of every feature.

    shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP_Value": shap_values[0]
    })


    # Absolute SHAP values determine which features
    # have the strongest influence.

    shap_df["Absolute_SHAP"] = (
        shap_df["SHAP_Value"].abs()
    )


    # --------------------------------------------------------
    # Select top five contributors
    # --------------------------------------------------------

    top_5 = (
        shap_df
        .nlargest(
            5,
            "Absolute_SHAP"
        )
        .copy()
    )


    # Return all relevant prediction information.

    return {
        "probability": wildfire_probability,
        "prediction": prediction,
        "shap_values": shap_df,
        "top_5": top_5,
        "input_df": input_df
    }


# ============================================================
# Demo scenarios
# ============================================================

# These predefined scenarios are controlled demonstration
# inputs rather than real-world observations.

scenarios = {

    "Lower-Risk Conditions": {

    "Precipitation_Average_30_Days": 12.0,

    "Specific_Humidity_Average_30_Days": 0.018,

    "Min_Relative_Humidity_Min_30_Days": 80.0,

    "Max_Relative_Humidity_Max_7_Days": 100.0,

    "Solar_Radiation_Average_7_Days": 70.0,

    "Min_Daily_Temperature_Min_14_Days": 0.0,

    "Max_Daily_Temperature_Max_30_Days": 12.0,

    "Wind_Speed_Average_30_Days": 0.5,

    "Vapor_Pressure_Deficit_Average_30_Days": 0.10,

    "Potential_Evapotranspiration_Average_30_Days": 0.5,

    "Solar_Radiation_Today": 70.0,

    "Min_Relative_Humidity_Today": 85.0,

    "Vapor_Pressure_Deficit_Today": 0.10,

    "Wind_Speed_Today": 0.5
},


    "Higher-Risk Conditions": {

        "Precipitation_Average_30_Days": 0.5,

        "Specific_Humidity_Average_30_Days": 0.006,

        "Min_Relative_Humidity_Min_30_Days": 18.0,

        "Max_Relative_Humidity_Max_7_Days": 55.0,

        "Solar_Radiation_Average_7_Days": 320.0,

        "Min_Daily_Temperature_Min_14_Days": 20.0,

        "Max_Daily_Temperature_Max_30_Days": 42.0,

        "Wind_Speed_Average_30_Days": 8.0,

        "Vapor_Pressure_Deficit_Average_30_Days": 3.5,

        "Potential_Evapotranspiration_Average_30_Days": 7.0,

        "Solar_Radiation_Today": 350.0,

        "Min_Relative_Humidity_Today": 18.0,

        "Vapor_Pressure_Deficit_Today": 4.0,

        "Wind_Speed_Today": 9.0
    }
}


# ============================================================
# Jupyter Interface — Input Card
# ============================================================

def create_input_card(feature, input_widgets):
    """
    Create a visual input card for one environmental feature.
    """

    (
        title,
        unit,
        description,
        explanation,
        min_value,
        max_value
    ) = input_definitions[feature]

    card = widgets.VBox(
        [
            widgets.HTML(
                value=f"""
                <div style="
                    margin-bottom:4px;
                ">
                    <strong style="font-size:14px;">
                        {title}
                    </strong>

                    <span style="
                        color:#777;
                        font-size:12px;
                    ">
                        ({unit})
                    </span>
                </div>
                """,
                layout=widgets.Layout(
                    height="24px"
                )
            ),

            widgets.HTML(
                value=f"""
                <div style="
                    color:#555;
                    font-size:12px;
                    line-height:1.4;
                ">
                    {description}
                </div>
                """,
                layout=widgets.Layout(
                    height="42px"
                )
            ),

            widgets.HTML(
                value=f"""
                <div style="
                    color:#777;
                    font-size:11px;
                    line-height:1.4;
                ">
                    {explanation}
                </div>
                """,
                layout=widgets.Layout(
                    height="50px"
                )
            ),

            # Flexible spacer pushes the range and input
            # to the same vertical position in every card.
            widgets.Box(
                layout=widgets.Layout(
                    flex="1 1 auto"
                )
            ),

            widgets.HTML(
                value=f"""
                <div style="
                    color:#999;
                    font-size:11px;
                    margin-bottom:5px;
                ">
                    Expected range: {min_value} – {max_value}
                </div>
                """,
                layout=widgets.Layout(
                    height="20px"
                )
            ),

            input_widgets[feature]
        ],

        layout=widgets.Layout(
            border="1px solid #ddd",
            border_radius="8px",
            padding="12px",
            margin="5px",
            width="100%",
            height="190px",
            box_sizing="border-box"
        )
    )

    return card


# ============================================================
# Jupyter Interface — Prediction Result
# ============================================================

def display_prediction_result(
    result,
    threshold,
    mode_name
):
    """
    Display probability, classification and SHAP results
    inside the Jupyter demo interface.
    """

    probability = result["probability"]

    prediction = result["prediction"]

    is_wildfire = prediction == 1


    # --------------------------------------------------------
    # Probability result
    # --------------------------------------------------------

    probability_html = widgets.HTML(
    value=f"""
    <div style="
        border:1px solid #ddd;
        border-radius:8px;
        padding:22px;
        margin-top:15px;
        width:100%;
        box-sizing:border-box;
        text-align:center;
    ">

        <div style="
            font-size:14px;
            color:#666;
            margin-bottom:6px;
        ">
            Wildfire Probability
        </div>

        <div style="
            font-size:52px;
            font-weight:700;
            line-height:1.1;
            margin-bottom:12px;
        ">
            {probability:.1%}
        </div>

        <div style="
            font-size:20px;
            font-weight:600;
            margin-bottom:10px;
        ">
            {
                "🔥 Wildfire"
                if is_wildfire
                else
                "🌳 No Wildfire"
            }
        </div>

        <div style="
            font-size:12px;
            color:#666;
            line-height:1.6;
        ">
            Mode: {mode_name}<br>
            Classification threshold: {threshold:.0%}
        </div>

    </div>
    """
)


    # --------------------------------------------------------
    # Prepare SHAP table
    # --------------------------------------------------------

    shap_table = result["top_5"].copy()

    feature_column = shap_table.columns[0]

    shap_column = shap_table.columns[1]

    shap_table = shap_table[
        [feature_column, shap_column]
    ].copy()

    shap_table.columns = [
        "Feature",
        "SHAP Value"
    ]


    # Use user-friendly feature names.

    shap_table["Feature"] = shap_table[
        "Feature"
    ].apply(
        lambda feature:
            display_names.get(
                feature,
                feature
            )
    )


    # --------------------------------------------------------
    # Format SHAP contribution direction
    # --------------------------------------------------------

    def format_contribution(value):

        if value > 0:

            return (
                '<span style="'
                'display:inline-block;'
                'width:32px;'
                'text-align:center;'
                'color:#d62728;'
                'font-size:24px;'
                'font-weight:bold;'
                'vertical-align:middle;'
                '">↑</span>'

                '<span style="vertical-align:middle;">'
                'Increases wildfire prediction'
                '</span>'
            )

        return (
            '<span style="'
            'display:inline-block;'
            'width:32px;'
            'text-align:center;'
            'color:#2ca02c;'
            'font-size:24px;'
            'font-weight:bold;'
            'vertical-align:middle;'
            '">↓</span>'

            '<span style="vertical-align:middle;">'
            'Decreases wildfire prediction'
            '</span>'
        )


    shap_table["Contribution"] = shap_table[
        "SHAP Value"
    ].apply(
        format_contribution
    )


    shap_table["SHAP Value"] = shap_table[
        "SHAP Value"
    ].round(3)


    shap_table = shap_table.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    shap_explanation = widgets.HTML(
        value="""
        <div style="
            margin-top:20px;
        ">

            <h4>
                Top 5 SHAP Contributors
            </h4>

            <p style="
                color:#666;
                font-size:14px;
                line-height:1.5;
            ">
                SHAP values explain how each feature influenced
                this individual prediction.
            </p>

            <p style="
                color:#888;
                font-size:13px;
                line-height:1.5;
            ">
                <span style="
                    color:#d62728;
                    font-size:18px;
                    font-weight:bold;
                ">↑</span>

                Increases wildfire prediction

                &nbsp;&nbsp;&nbsp;

                <span style="
                    color:#2ca02c;
                    font-size:18px;
                    font-weight:bold;
                ">↓</span>

                Decreases wildfire prediction
            </p>

            <p style="
                color:#888;
                font-size:13px;
                line-height:1.5;
            ">
                SHAP values represent contributions in the
                model's log-odds space and should not be
                interpreted as direct percentage-point changes
                in wildfire probability.
            </p>

        </div>
        """
    )


    # --------------------------------------------------------
    # Convert table to HTML
    # --------------------------------------------------------

    table_html = shap_table.to_html(
        index=False,
        escape=False
    )


    # --------------------------------------------------------
    # Apply table styling
    # --------------------------------------------------------

    table_html = table_html.replace(
        "<table",
        """
        <table style="
            width:100%;
            border-collapse:collapse;
            margin-top:10px;
            font-size:13px;
        "
        """
    )


    table_html = table_html.replace(
        "<th>",
        """
        <th style="
            padding:9px 12px;
            background:#eeeeee;
            text-align:left;
            vertical-align:middle;
        ">
        """
    )


    table_html = table_html.replace(
        "<td>",
        """
        <td style="
            padding:9px 12px;
            text-align:left;
            vertical-align:middle;
        ">
        """
    )


    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display(probability_html)

    display(shap_explanation)

    display(
        HTML(table_html)
    )


# ============================================================
# Jupyter Interface — Complete Demo
# ============================================================

def create_demo_interface():
    """
    Create and display the complete Jupyter demo interface.

    The interface allows the user to:
    - enter the 14 environmental model features;
    - select a prediction threshold;
    - run the model;
    - inspect wildfire probability;
    - inspect the classification;
    - inspect the top five SHAP contributors.
    """

    # --------------------------------------------------------
    # Create fresh input widgets
    # --------------------------------------------------------

    input_widgets = {}

    for feature in features:

        (
            title,
            unit,
            description,
            explanation,
            min_value,
            max_value
        ) = input_definitions[feature]


        # Use the midpoint of the expected range as
        # the default value.

        default_value = (
            min_value + max_value
        ) / 2


        input_widgets[feature] = widgets.FloatText(
            value=default_value,
            layout=widgets.Layout(
                width="100%"
            )
        )


    # --------------------------------------------------------
    # Create input cards
    # --------------------------------------------------------

    input_cards = [
        create_input_card(
            feature,
            input_widgets
        )
        for feature in features
    ]


    # --------------------------------------------------------
    # Arrange cards into aligned rows
    # --------------------------------------------------------

    input_rows = []

    for i in range(
        0,
        len(input_cards),
        2
    ):

        row = widgets.HBox(
            [
                input_cards[i],
                input_cards[i + 1]
            ],

            layout=widgets.Layout(
                width="100%",
                align_items="stretch"
            )
        )

        input_rows.append(row)


    # --------------------------------------------------------
    # Environmental conditions section
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Predefined scenario autofill buttons
    # --------------------------------------------------------

    lower_risk_button = widgets.Button(
        description="🌳 Low Wildfire Risk Example",
        layout=widgets.Layout(
            width="220px",
            height="38px"
        )
    )

    higher_risk_button = widgets.Button(
        description="🔥 High Wildfire Risk Example",
        layout=widgets.Layout(
            width="220px",
            height="38px"
        )
    )

    scenario_buttons = widgets.HBox(
        [
            lower_risk_button,
            higher_risk_button
        ],
        layout=widgets.Layout(
            width="100%",
            justify_content="center",
            gap="12px",
            margin="0 0 12px 0"
        )
    )

    def fill_scenario(scenario_name):
        """Fill all input widgets with a predefined scenario."""

        scenario = scenarios[scenario_name]

        for feature in features:
            input_widgets[feature].value = scenario[feature]

    lower_risk_button.on_click(
        lambda button: fill_scenario("Lower-Risk Conditions")
    )

    higher_risk_button.on_click(
        lambda button: fill_scenario("Higher-Risk Conditions")
    )

    # --------------------------------------------------------
    # Environmental conditions section
    # --------------------------------------------------------

    input_interface = widgets.VBox(
        [
            widgets.HTML(
    value="""
    <div style="
        text-align:center;
        margin-bottom:15px;
    ">
        <h3 style="
            margin-bottom:6px;
        ">
            Environmental Conditions
        </h3>

        <p style="
            color:#666;
            font-size:13px;
            margin-top:0;
        ">
            Enter the environmental conditions manually, or use a predefined
            scenario to autofill the inputs.
        </p>
    </div>
    """
),

            scenario_buttons,

            *input_rows

        ],

        layout=widgets.Layout(
            width="100%"
        )
    )


    # ========================================================
    # Header
    # ========================================================

    header_path = BASE_DIR / "header.jpg"


    with open(
        header_path,
        "rb"
    ) as f:

        header_image = widgets.Image(
            value=f.read(),
            format="jpg",
            layout=widgets.Layout(
                width="1000px",
                height="auto"
            )
        )


    header_container = widgets.HBox(
        [header_image],

        layout=widgets.Layout(
            width="100%",
            justify_content="center",
            margin="0 0 20px 0"
        )
    )


    # --------------------------------------------------------
    # Title and instructions
    # --------------------------------------------------------

    demo_header = widgets.HTML(
        value="""
        <div style="
            max-width:1000px;
            margin:0 auto 25px auto;
            text-align:center;
        ">

            <h1 style="
                font-size:30px;
                margin:0 0 8px 0;
            ">
                Wildfire Ignition Risk Prediction
            </h1>

            <p style="
                font-size:16px;
                color:#555;
                margin:0 0 18px 0;
            ">
                Estimate the probability of wildfire ignition
                from recent environmental conditions using a
                machine learning model.
            </p>

            <div style="
                text-align:left;
                background:#f7f7f7;
                border-radius:8px;
                padding:15px 20px;
                font-size:14px;
                line-height:1.6;
                color:#555;
            ">

                <strong>
                    How to use the demo
                </strong>

                <ol style="
                    margin:8px 0 0 20px;
                    padding:0;
                ">

                    <li>
                        Enter the environmental conditions manually, or use
                        one of the two predefined scenarios to automatically
                        fill in the inputs.
                    </li>

                    <li>
                        Select a prediction mode.
                    </li>

                    <li>
                        Click <strong>Run Prediction</strong>
                        to obtain the estimated wildfire
                        probability.
                    </li>

                    <li>
                        Review the Top 5 SHAP contributors to
                        understand which features influenced
                        the prediction.
                    </li>

                </ol>

            </div>

        </div>
        """
    )


    # ========================================================
    # Prediction controls
    # ========================================================

    mode_selector = widgets.ToggleButtons(
        options=[
            (
                "Balanced Mode",
                balanced_threshold
            ),
            (
                "High-Sensitivity Mode",
                high_sensitivity_threshold
            )
        ],

        description="",

        tooltips=[
            "Uses a 0.50 classification threshold.",
            "Uses a 0.35 classification threshold to prioritise wildfire detection."
        ],

        layout=widgets.Layout(
            width="auto",
            justify_content="center"
        ),

        style={
            "button_width": "210px"
        }
    )


    run_button = widgets.Button(
        description="Run Prediction",
        button_style="primary",
        icon="play",

        layout=widgets.Layout(
            width="250px",
            height="45px"
        )
    )


    prediction_output = widgets.Output()


    # --------------------------------------------------------
    # Prediction callback
    # --------------------------------------------------------

    def run_prediction(button):

        prediction_output.clear_output(
            wait=True
        )


        # Collect the 14 environmental inputs.

        input_data = {
            feature: input_widgets[feature].value
            for feature in features
        }


        # Get selected threshold.

        selected_threshold = (
            mode_selector.value
        )


        # Determine mode name.

        if selected_threshold == balanced_threshold:

            mode_name = "Balanced Mode"

        else:

            mode_name = "High-Sensitivity Mode"


        # Run prediction.

        result = predict_wildfire(
            input_data,
            threshold=selected_threshold
        )


        # Display prediction result.

        with prediction_output:

            display_prediction_result(
                result,
                selected_threshold,
                mode_name
            )


    # Connect button to callback.

    run_button.on_click(
        run_prediction
    )


    # --------------------------------------------------------
    # Prediction mode section
    # --------------------------------------------------------

    mode_section = widgets.VBox(
        [
            widgets.HTML(
                value="""
                <h3 style="
                    text-align:center;
                    margin:10px 0 12px 0;
                ">
                    Prediction Mode
                </h3>
                """
            ),

            mode_selector,

            widgets.Box(
                layout=widgets.Layout(
                    height="15px"
                )
            ),

            run_button

        ],

        layout=widgets.Layout(
            width="100%",
            align_items="center",
            padding="5px 0 20px 0"
        )
    )


        # ========================================================
    # Complete interface
    # ========================================================

    complete_interface = widgets.VBox(
        [
            header_container,
            demo_header,
            input_interface,
            mode_section,
            prediction_output
        ],

        layout=widgets.Layout(
            width="100%",
            max_width="1000px",
            margin="0 auto"
        )
    )


    # Display only the complete interface.

    display(
        complete_interface
    )


    # Return the interface so it can also be reused
    # programmatically if needed.

    return complete_interface


# ============================================================
# Interactive Terminal Demo
# ============================================================
# ============================================================

def run_interactive_demo():
    """
    Run the interactive terminal version of the demo.

    The user enters the environmental conditions once and
    can then evaluate the same conditions using different
    classification thresholds.
    """

    # --------------------------------------------------------
    # INPUT 1 — Environmental conditions
    # --------------------------------------------------------

    print("=" * 65)

    print(
        "       INPUT 1 — ENVIRONMENTAL CONDITIONS"
    )

    print("=" * 65)


    print(
        "\nEnter the environmental conditions used by the model."
    )


    print(
        "These values will be stored and can be reused with "
        "different prediction modes.\n"
    )


    # Dictionary used to store user input.

    user_input = {}


    # --------------------------------------------------------
    # Collect the 14 environmental variables
    # --------------------------------------------------------

    for feature in features:

        (
            label,
            unit,
            description,
            explanation,
            min_value,
            max_value
        ) = input_definitions[feature]


        # Keep asking until a valid value is entered.

        while True:

            print(
                f"{label} ({unit})"
            )

            print(
                description
            )

            print(
                explanation
            )

            print(
                f"Expected range: "
                f"{min_value} – {max_value}"
            )


            try:

                value = float(
                    input("Value: ")
                )


                # Validate the value against the
                # accepted range.

                if min_value <= value <= max_value:

                    user_input[feature] = value

                    print()

                    break


                print(
                    f"\nValue must be between "
                    f"{min_value} and {max_value}.\n"
                )


            except ValueError:

                print(
                    "\nPlease enter a numeric value.\n"
                )


    # --------------------------------------------------------
    # Confirm input collection
    # --------------------------------------------------------

    print("=" * 65)

    print(
        "Environmental conditions successfully stored."
    )

    print("=" * 65)


    # --------------------------------------------------------
    # Prediction mode selection
    # --------------------------------------------------------

    while True:

        print("\n")

        print("=" * 65)

        print(
            "             INPUT 2 — PREDICTION MODE"
        )

        print("=" * 65)


        print(
            "\nSelect how you want the model to classify "
            "the prediction:\n"
        )


        # Balanced mode.

        print("1. Balanced Mode")

        print(
            f"   Threshold: "
            f"{balanced_threshold:.0%}"
        )

        print(
            "   More balanced between detecting wildfires "
            "and avoiding false alarms."
        )


        # High-sensitivity mode.

        print("\n2. High-Sensitivity Mode")

        print(
            f"   Threshold: "
            f"{high_sensitivity_threshold:.0%}"
        )

        print(
            "   Prioritizes detecting wildfire events, "
            "accepting more false alarms."
        )


        # Exit option.

        print("\n3. Exit")


        mode = input(
            "\nSelect mode (1/2/3): "
        ).strip()


        # ----------------------------------------------------
        # Select threshold
        # ----------------------------------------------------

        if mode == "1":

            selected_mode = "Balanced Mode"

            threshold = balanced_threshold


        elif mode == "2":

            selected_mode = "High-Sensitivity Mode"

            threshold = high_sensitivity_threshold


        elif mode == "3":

            print(
                "\nExiting wildfire prediction demo."
            )

            break


        else:

            print(
                "\nPlease select 1, 2, or 3."
            )

            continue


        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        result = predict_wildfire(
            user_input,
            threshold
        )


        # Extract results.

        wildfire_probability = result["probability"]

        prediction = result["prediction"]

        top_5 = result["top_5"]


        # ----------------------------------------------------
        # Display prediction result
        # ----------------------------------------------------

        print("\n")

        print("=" * 65)

        print(
            "                    PREDICTION RESULT"
        )

        print("=" * 65)


        print(
            f"\nPrediction mode: "
            f"{selected_mode}"
        )


        print(
            f"Classification threshold: "
            f"{threshold:.0%}"
        )


        print(
            f"\nWildfire probability: "
            f"{wildfire_probability:.2%}"
        )


        # Display classification.

        if prediction == 1:

            print(
                "\n🔥  WILDFIRE RISK"
            )

        else:

            print(
                "\n✅  NO WILDFIRE RISK"
            )


        # ----------------------------------------------------
        # Display top five SHAP contributors
        # ----------------------------------------------------

        print("\n")

        print(
            "Top 5 factors influencing this prediction"
        )

        print("-" * 65)


        for rank, (_, row) in enumerate(
            top_5.iterrows(),
            start=1
        ):

            feature = row["Feature"]

            shap_value = row["SHAP_Value"]


            # Retrieve original user input.

            value = user_input[feature]


            # Retrieve display name and unit.

            name = display_names[feature]

            unit = units[feature]


            # Determine SHAP direction.

            if shap_value > 0:

                effect = (
                    "↑ Increases wildfire probability"
                )

            else:

                effect = (
                    "↓ Decreases wildfire probability"
                )


            print(
                f"\n{rank}. {name}"
            )


            print(
                f"   Input value: "
                f"{value:.3f} {unit}"
            )


            print(
                f"   Log-odds contribution: "
                f"{shap_value:+.3f}"
            )


            print(
                f"   Effect: {effect}"
            )


        # ----------------------------------------------------
        # SHAP interpretation note
        # ----------------------------------------------------

        print("\n")

        print(
            "SHAP values represent the direction and magnitude "
            "of each feature's contribution to the model's "
            "log-odds output."
        )


        print(
            "They are not percentage-point changes in "
            "wildfire probability."
        )


        # ----------------------------------------------------
        # Allow another prediction
        # ----------------------------------------------------

        print("\n")

        print("-" * 65)


        print(
            "The same environmental conditions are still stored."
        )


        print(
            "You can select another prediction mode "
            "without re-entering the data."
        )


    print(
        "\nDemo finished."
    )


# ============================================================
# Run terminal demo only when executed directly
# ============================================================

# When this file is imported by the notebook, the terminal
# application is not launched automatically.
#
# Running the file directly from Terminal launches the
# interactive terminal demo.

if __name__ == "__main__":

    run_interactive_demo()
