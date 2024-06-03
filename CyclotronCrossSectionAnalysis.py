import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import re
from PyPDF2 import PdfReader 
from scipy.interpolate import interp1d
pd.set_option('display.max_columns', 500)

irradiation_time =       600 # seconds
EOB =                    '05/26/2022 7:37' # %m/%d/%Y %H:%M
mass_foil_Cu = 17.1 # mg
mass_foil_V  = 64.5
requested_current = 5 # uA

dilution_corrections = {
    '62Zn': 8,  # Example values
    '65Zn': 5,
    '49Cr': 10,
    '51Cr': 10,
    '47Sc': 10
}

isotope_parameters = {
    '62Zn': {'half_life': 33094.80, 'lambda_uncertainty': 1.5E-5},
    '65Zn': {'half_life': 21075552.00, 'lambda_uncertainty': 9E-8},
    '49Cr': {'half_life': 2532.0, 'lambda_uncertainty': 1E-4},
    '51Cr': {'half_life': 2393625.60, 'lambda_uncertainty': 2.4E-7},
    '47Sc': {'half_life': 289370.88, 'lambda_uncertainty': 6E-6}
}

df_IsotopeInformation = pd.read_csv('./62,65Zn 49,51Cr 47Sc Isotope Information.csv')

def compute_activity(net_peak_area, decay_constant, live_time, efficiency, branching_ratio):
    """
    Computes the activity in microcuries (uCi).

    Parameters:
    - net_peak_area: Net peak area.
    - decay_constant: Decay constant (lambda) in inverse seconds.
    - live_time: Live time in seconds.
    - efficiency: Detection efficiency.
    - branching_ratio: Branching ratio of the gamma ray.

    Returns:
    - Activity in microcuries (uCi).
    """
    # Calculate the denominator part of the equation
    denominator = (1 - np.exp(-decay_constant * live_time)) * efficiency * branching_ratio * 37000

    numerator = net_peak_area * decay_constant
    
    # Compute activity
    activity_uCi = numerator / denominator
    
    return activity_uCi

def adjusted_activity(activity, dilution_correction):
    return activity*dilution_correction

def calculate_activity_at_EOB(time_difference, average_adjusted_activity, λ):
    """
    Calculate the activity at end of bombardment.

    Parameters:
    average_adjusted_activity (float): The average adjusted activity.
    decay_constant (float): The decay constant (λ).

    Returns:
    float: The activity at the end of bombardment.
    """
    
    # Adjust activity using the exponential decay formula with NumPy's exp function
    activity_at_EOB = average_adjusted_activity * np.exp(time_difference * λ)
    
    return activity_at_EOB

def process_pdf_extract_data(pdf_path, EOB):
    # Define the regex pattern for measurement time inside the function
    measurement_time_re = r"Acquisition Started\s*:\s*(\d+/\d+/\d+\s+\d+:\d+:\d+\s+[AP]M)"

    # Parse the EOB datetime string
    EOB_datetime = datetime.strptime(EOB.strip(), "%m/%d/%Y %H:%M")

    # Create a PDF reader object
    reader = PdfReader(pdf_path)

    # Live and Dead Time extraction, along with Measurement Time from the first page
    page_text = reader.pages[0].extract_text()  # Accessing the first page

    live_time_re = r"Live Time\s+:\s+([\d.]+) seconds"
    real_time_re = r"Real Time\s+:\s+([\d.]+) seconds"

    live_time_match = re.search(live_time_re, page_text)
    real_time_match = re.search(real_time_re, page_text)
    measurement_time_match = re.search(measurement_time_re, page_text)

    live_time = float(live_time_match.group(1)) if live_time_match else None
    real_time = float(real_time_match.group(1)) if real_time_match else None
    measurement_time = datetime.strptime(measurement_time_match.group(1), "%m/%d/%Y %I:%M:%S %p") if measurement_time_match else None
    dead_time = real_time - live_time if live_time and real_time else None

    # Calculate the difference in time in seconds between EOB and measurement time
    if measurement_time:
        time_difference_seconds = (measurement_time - EOB_datetime).total_seconds()
    else:
        time_difference_seconds = None

    # Peak Efficiency Report extraction from the third page
    PeakEfficiencyReport = reader.pages[2].extract_text()  # Accessing the third page directly
    lines = PeakEfficiencyReport.split('\n')[9:]  # Skip header lines

    column_headers = ['Class', 'Peak No.', 'Energy (keV)', 'Net Peak Area', 'Net Area Uncertainty', 'Peak Efficiency', 'Efficiency Uncertainty']

    data = []
    for line in lines:
        if line.strip():
            columns = re.split(r'\s{2,}', line.strip())
            if len(columns) == len(column_headers):
                data.append(columns)

    df_PeakEfficiencyReport = pd.DataFrame(data, columns=column_headers)

    # Since the Live/Dead Time and Measurement Time data applies to all rows
    repeated_values = [{
        "Live Time (seconds)": live_time, 
        "Real Time (seconds)": real_time, 
        "Dead Time (seconds)": dead_time,
        "Measurement Time": measurement_time,
        "Time Difference (seconds)": time_difference_seconds  # Adding the new calculated field
    }] * len(df_PeakEfficiencyReport)
    df_LiveDeadTime = pd.DataFrame(repeated_values)

    # Concatenating DataFrames
    df_HPGe = pd.concat([df_PeakEfficiencyReport.reset_index(drop=True), df_LiveDeadTime], axis=1)
    
    return df_HPGe

def integrate(df_HPGe, df_IsotopeInformation, irradiation_time, mass_foil):
    # Convert necessary columns to numeric
    df_HPGe['Energy (keV)'] = pd.to_numeric(df_HPGe['Energy (keV)'], errors='coerce')
    df_IsotopeInformation['Gamma Rays (keV)'] = pd.to_numeric(df_IsotopeInformation['Gamma Rays (keV)'], errors='coerce')
    
    # Merge and clean data
    new_rows = []
    for index, row in df_HPGe.iterrows():
        energy = row['Energy (keV)']
        match_found = False
        for _, isotope_row in df_IsotopeInformation.iterrows():
            if abs(energy - isotope_row['Gamma Rays (keV)']) <= 2:
                match_found = True
                new_row = {**row.to_dict(), **isotope_row.to_dict(), 'Matching Gamma Peak (keV)': isotope_row['Gamma Rays (keV)']}
                new_rows.append(new_row)
                break
        if not match_found:
            new_row = row.to_dict()
            new_row.update({'Matching Gamma Peak (keV)': pd.NA, 'Half life (s)': pd.NA, 'Branching Ratio': pd.NA})
            new_rows.append(new_row)
    df_combined = pd.DataFrame(new_rows).dropna(subset=['Matching Gamma Peak (keV)'])
    
    # Append constants
    df_combined['Irradiation Time (s)'] = irradiation_time
    df_combined['Foil Mass (mg)'] = mass_foil
    
    # Ensure all columns are numeric, except for exclusions
    exclude_columns = ['Class', 'Peak No.', 'Isotope', 'Measurement Time']
    for col in df_combined.columns.difference(exclude_columns):
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    # Map dilution corrections to isotopes and compute activities
    df_combined['Dilution Correction'] = df_combined['Isotope'].map(dilution_corrections)
    
    return df_combined

def compute(df_computed):
    # Compute activities and adjust
    df_computed['Activity (uCi)'] = df_computed.apply(
        lambda row: compute_activity(
            net_peak_area=row['Net Peak Area'], 
            decay_constant=np.log(2) / row['Half life (s)'], 
            live_time=row['Live Time (seconds)'], 
            efficiency=row['Peak Efficiency'], 
            branching_ratio=row['Branching Ratio']
        ), axis=1
    )
    
    # Adjust activity based on the mapped dilution correction
    df_computed['Adjusted Activity (uCi)'] = df_computed.apply(
        lambda row: adjusted_activity(row['Activity (uCi)'], row['Dilution Correction']),
        axis=1
    )
    
    # Calculate average activity and append as a constant column
    average_activity = df_computed['Adjusted Activity (uCi)'].mean()
    df_computed['Average Adjusted Activity (uCi)'] = average_activity
    
    # Calculate activity at EOB for each row
    df_computed['Activity at EOB (uCi)'] = df_computed.apply(
        lambda row: calculate_activity_at_EOB(
            time_difference=row['Time Difference (seconds)'],
            average_adjusted_activity=row['Average Adjusted Activity (uCi)'],
            λ=np.log(2) / row['Half life (s)']
        ), axis=1
    )

    return df_computed

def ratio_theory(σZn62, σZn65, irradiation_time):
    """
    Ratio Theory Equation
    (σZn62 * (1 - exp(-λ62 * 600))) / (σZn65 * (1 - exp(-λ65 * 600)))
    
    σZn62: σ for 62Zn
    σZn65: σ for 65Zn
    λ62: λ for Zn62
    λ65: λ for Zn65
    return: Calculated ratios for t seconds of irradiation time
    """
    λ62 = 2.09443E-05
    λ65 = 3.28887E-08
    ratio = (σZn62 * (1 - np.exp(-λ62 * irradiation_time))) / (σZn65 * (1 - np.exp(-λ65 * irradiation_time)))
    return ratio

# Function to output Energy, σZn62, and σZn65 for a given σZn62/σZn65
def get_values_for_ratio(ratio):
    energy = energy_interp(ratio)
    sigma62 = sigma62_interp(ratio)
    sigma65 = sigma65_interp(ratio)
    return energy, sigma62, sigma65

def compute_correction_factor_and_uncertainty(isotope, livetime, isotope_parameters):
    if isotope not in isotope_parameters:
        return (np.nan, np.nan)  # Return NaNs for both values if isotope is not found
    
    # Retrieve half-life and lambda uncertainty for the isotope
    half_life = isotope_parameters[isotope]['half_life']
    lambda_uncertainty = isotope_parameters[isotope]['lambda_uncertainty']
    
    # Ensure half_life and livetime are not zero to avoid division by zero
    if half_life == 0 or livetime == 0:
        return (np.nan, np.nan)  # Return NaNs for both values
    
    # Calculate lambda (decay constant) and correction factor
    λ = np.log(2) / half_life
    correction_factor = (1 - np.exp(-λ * livetime)) / (λ * livetime)
    
    # Calculate the correction factor uncertainty
    correction_factor_uncertainty = lambda_uncertainty * (1 - np.exp(-λ * livetime)) / (λ * livetime)
    
    return correction_factor, correction_factor_uncertainty

def activity_uncertainty(adjusted_activity, correction_factor, correction_factor_uncertainty, 
                         branching_ratio, 
                         branching_ratio_uncertainty, 
                         net_peak, net_peak_uncertainty,
                         efficiency, efficiency_uncertainty):
    term1 = (correction_factor_uncertainty/correction_factor)**2
    term2 = (branching_ratio_uncertainty/branching_ratio)**2
    term3 = (net_peak_uncertainty/net_peak)**2
    term4 = (efficiency_uncertainty/efficiency)**2
    return adjusted_activity*np.sqrt(term1 + term2 + term3 + term4)   

def add_average_uncertainty(df):
    """
    Adds a new column to the DataFrame with the average 'Adjusted Activity Uncertainty (uCi)'
    for each isotope.
    """
    # Calculate the average uncertainty for each isotope
    averages = df.groupby('Isotope')['Adjusted Activity Uncertainty (uCi)'].transform('mean')
    
    # Store the average in a new column
    df['Average Adjusted Activity Uncertainty (uCi)'] = averages
    
    return df

def calculate_eob_uncertainty(half_life, activity_at_EOB, λ_uncertainty, average_adjusted_activity_uncertainty, average_adjusted_activity, time_difference):
    """
    Calculate the End of Bombardment (EOB) uncertainty.

    Parameters:
    activity_at_eob (float): The activity at the end of bombardment.
    λ_uncertainty (float): The uncertainty in the exponential decay constant (λ).
    adjusted_activity_uncertainty (float): The uncertainty in the adjusted activity.
    adjust_activity (float): The adjusted activity itself.
    time_difference (float): The time difference in the same units as the decay constant (λ).

    Returns:
    float: The EOB uncertainty.
    """

    λ = np.log(2) / half_life
    
    # Calculate the decay factor
    decay_factor = np.exp(λ * time_difference)
    
    # Calculate the first part of the uncertainty equation
    decay_factor_uncertainty = decay_factor * λ_uncertainty

    # Term 1
    term1 = (decay_factor_uncertainty / decay_factor) ** 2
    
    # Calculate the second part of the uncertainty equation
    term2 = (average_adjusted_activity_uncertainty / average_adjusted_activity) ** 2
    
    # Calculate the total EOB uncertainty
    eob_uncertainty = activity_at_EOB * np.sqrt(term1 + term2)
    
    return eob_uncertainty

def compute_energy_uncertainty(df):
    # Filter rows for Zn62 and Zn65
    zn62_row = df[df['Isotope'] == '62Zn'].iloc[0]
    zn65_row = df[df['Isotope'] == '65Zn'].iloc[0]
    
    # Compute terms inside the square root for Zn62 and Zn65
    term_zn62 = (zn62_row['Activity at EOB Uncertainty (uCi)'] / zn62_row['Activity at EOB (uCi)']) ** 2
    term_zn65 = (zn65_row['Activity at EOB Uncertainty (uCi)'] / zn65_row['Activity at EOB (uCi)']) ** 2
    
    energy = zn62_row['Energy (MeV)']
    
    # Compute the energy uncertainty
    energy_uncertainty = energy * np.sqrt(term_zn62 + term_zn65)
    
    return energy_uncertainty

def compute_flux(activity_62Zn_at_EOB, mass_foil_Cu, cross_section_62Zn, irradiation_time):
    """
    Computes the neutron flux based on the provided parameters.

    Parameters:
    - activity_62Zn: Activity of 62Zn in microcuries (uCi).
    - density: Density of the material in g/cm³.
    - AMU: Atomic mass unit of the material.
    - mass_foil: Mass of the foil in mg
    - radius: Radius of the foil in cm.
    - cross_section_62Zn: Cross section of 62Zn in barns.
    - irradiation_time: Irradiation time in seconds.
    - half_life: Half-life of 62Zn in seconds.

    Returns:
    - Flux: Computed proton flux.
    """
    avogadros_number = 6.022E23  # mol^-1
    pi = np.pi
    radius = 0.5
    half_life = 33094.8
    density = 8.96
    AMU = 63.546
    
    thickness_um = (mass_foil_Cu * 10000) / (1000 * density * pi * radius**2)

    thickness_cm = thickness_um / 10000

    atoms_per_cm2 = (avogadros_number * density * thickness_cm ) / AMU
    
    # Compute the correction factor for decay
    decay_correction = 1 - np.exp(-(irradiation_time * np.log(2)) / half_life)

    # Calculate the flux
    flux = (activity_62Zn_at_EOB * 37000) / (atoms_per_cm2 * cross_section_62Zn*1E-27 * decay_correction)
    return flux

def compute_flux_uncertainty(Zn62_EOB, Zn62_EOB_uncertainty, flux):
    """
    Computes the neutron flux based on the provided parameters.

    Parameters:
    - activity_62Zn: Activity of 62Zn in microcuries (uCi).
    - density: Density of the material in g/cm³.
    - AMU: Atomic mass unit of the material.
    - mass_foil: Mass of the foil in mg
    - radius: Radius of the foil in cm.
    - cross_section_62Zn: Cross section of 62Zn in barns.
    - irradiation_time: Irradiation time in seconds.
    - half_life: Half-life of 62Zn in seconds.

    Returns:
    - Flux: Computed proton flux.
    """
    half_life = 32312.0
    lambda_uncertainty = 1.5E-5
    lambda_ = np.log(2) / half_life
    
    term1 = (lambda_uncertainty / lambda_) **2
    term2 = (Zn62_EOB_uncertainty / Zn62_EOB) **2
    
    # Calculate the flux
    flux_uncertainty = flux * np.sqrt(term1 + term2)
    return flux_uncertainty

def cross_section_for_specific_isotopes(row):
    # Specify the isotopes for which you want to compute the function
    specific_isotopes = ['49Cr', '51Cr', '47Sc']

    pi = np.pi
    density = 6.11 # g/cm^3
    radius = 0.5 
    avogadros_number = 6.022E23
    AMU = 50.9415
    
    thickness_um = (mass_foil_V * 10000) / (1000 * density * pi * radius**2)
    thickness_cm = thickness_um / 10000
    
    atoms_per_cm2 = (avogadros_number * density * thickness_cm ) / AMU

    decay_correction = 1 - np.exp(-(row['Irradiation Time (s)'] * np.log(2)) / row['Half life (s)'])
    # Check if the isotope in the current row is one of the specific isotopes
    if row['Isotope'] in specific_isotopes:
        # Compute and return the desired value for specific isotopes
        cross_section = (row['Activity at EOB (uCi)'] * 37000) / (atoms_per_cm2 * row['Flux (proton/s)'] * decay_correction)
        return cross_section
    else:
        # Return NaN for isotopes that do not match the criteria
        return np.nan

def cross_section_uncertainty(half_life, lambda_uncertainty, time_difference, flux, flux_uncertainty, cross_section):
    """
    Calculate the End of Bombardment (EOB) uncertainty.

    Parameters:
    activity_at_eob (float): The activity at the end of bombardment.
    λ_uncertainty (float): The uncertainty in the exponential decay constant (λ).
    adjusted_activity_uncertainty (float): The uncertainty in the adjusted activity.
    adjust_activity (float): The adjusted activity itself.
    time_difference (float): The time difference in the same units as the decay constant (λ).

    Returns:
    float: The Cross Section uncertainty in mb.
    """

    λ = np.log(2) / half_life
    
    # Calculate the decay factor
    decay_factor = np.exp(λ * time_difference)
    
    # Calculate decay factor uncertainty
    decay_factor_uncertainty = decay_factor * lambda_uncertainty

    # Term 1
    term1 = (flux_uncertainty / flux) **2
    # Term 2
    term2 = (lambda_uncertainty / λ) ** 2
    # Term 3
    term3 = (decay_factor_uncertainty / decay_factor) ** 2
    
    # Calculate the total EOB uncertainty
    cross_section_uncertainty = cross_section * np.sqrt(term1 + term2 + term3)
    
    return cross_section_uncertainty

def color_rows_by_isotope(s):
    # Define colors for each isotope
    colors = {
        '62Zn': 'background-color: #d9faed',
        '65Zn': 'background-color: #70dbb0',
        '49Cr': 'background-color: #fca9b0',
        '51Cr': 'background-color: #fa7a85',
        '47Sc': 'background-color: #eac5fc'
        # Define more colors for other isotopes as needed
    }
    # Apply the color for the whole row based on the 'Isotope' column's value
    return [colors.get(s['Isotope'], '')] * len(s)



pdf_path_CuShort = r'./CuFoil1_A_p10_10min_scond count.PDF'  # Replace with your actual PDF file path
df_HPGe_CuShort = process_pdf_extract_data(pdf_path_CuShort, EOB)
df_CuShort = integrate(df_HPGe_CuShort, df_IsotopeInformation, irradiation_time, mass_foil_Cu)
df_CuShortFiltered = df_CuShort[df_CuShort['Isotope'] == '62Zn']
df_CuShort = df_CuShortFiltered.copy()
df_CuShortComputed = compute(df_CuShort)

pdf_path_CuLong = r'./CuFoil1_A_default_10hr_FourthCount.PDF'  # Replace with your actual PDF file path
df_HPGe_CuLong = process_pdf_extract_data(pdf_path_CuLong, EOB)
df_CuLong = integrate(df_HPGe_CuLong, df_IsotopeInformation, irradiation_time, mass_foil_Cu)
df_CuLongFiltered = df_CuLong[df_CuLong['Isotope'] == '65Zn']
df_CuLong = df_CuLongFiltered.copy()
df_CuLongComputed = compute(df_CuLong)

pdf_path_V = r'./VFoil1_A_.PDF'  # Replace with your actual PDF file path
df_HPGe_V = process_pdf_extract_data(pdf_path_V, EOB)
df_V = integrate(df_HPGe_V, df_IsotopeInformation, irradiation_time, mass_foil_V)

df_49CrFiltered = df_V[df_V['Isotope'] == '49Cr']
df_49Cr = df_49CrFiltered.copy()
df_49CrComputed = compute(df_49Cr)

df_51CrFiltered = df_V[df_V['Isotope'] == '51Cr']
df_51Cr = df_51CrFiltered.copy()
df_51CrComputed = compute(df_51Cr)

df_47ScFiltered = df_V[df_V['Isotope'] == '47Sc']
df_47Sc = df_47ScFiltered.copy()
df_47ScComputed = compute(df_47Sc)

df_All = pd.concat([df_CuShortComputed, df_CuLongComputed, df_49CrComputed, df_51CrComputed, df_47ScComputed], axis=0).reset_index(drop=True)



file_path = 'RatioTheoryInputZn6265.csv'

# Read the CSV file into a DataFrame
df_RatioTheoryInformation = pd.read_csv(file_path)

df_RatioTheoryInformation['Irradiation Time (s)'] = irradiation_time

# Make a copy of the original DataFrame
df_RatioTheory = df_RatioTheoryInformation.copy()

# Apply the compute_activity function to each row of the copied DataFrame
df_RatioTheory['σZn62/σZn65'] = df_RatioTheory.apply(lambda row: ratio_theory(
    σZn62 = row['62Zn (mbarn)'],
    σZn65 = row['65Zn (mbarn)'],
    irradiation_time = row['Irradiation Time (s)']
), axis=1)

# Find the index of the first occurrence of '62Zn' and '65Zn' in the 'Isotope' column
index_62Zn = df_All[df_All['Isotope'] == '62Zn'].index[0]
index_65Zn = df_All[df_All['Isotope'] == '65Zn'].index[0]

# Retrieve the 'Activity at EOB (uCi)' for the first occurrence of '62Zn' and '65Zn'
activity_62Zn = df_All.loc[index_62Zn, 'Activity at EOB (uCi)']
activity_65Zn = df_All.loc[index_65Zn, 'Activity at EOB (uCi)']

# Divide the 'Activity at EOB (uCi)' of '62Zn' by that of '65Zn'
ratio = activity_62Zn / activity_65Zn

# Interpolating function for Energy based on σZn62/σZn65 ratio
energy_interp = interp1d(df_RatioTheory['σZn62/σZn65'], df_RatioTheory['Energy (meV)'], kind='linear', fill_value="extrapolate")

# Interpolating functions for σZn62 and σZn65 based on σZn62/σZn65 ratio
sigma62_interp = interp1d(df_RatioTheory['σZn62/σZn65'], df_RatioTheory['62Zn (mbarn)'], kind='linear', fill_value="extrapolate")
sigma65_interp = interp1d(df_RatioTheory['σZn62/σZn65'], df_RatioTheory['65Zn (mbarn)'], kind='linear', fill_value="extrapolate")

interpolated_ratio_theory = get_values_for_ratio(ratio)
interpolated_energy = float(interpolated_ratio_theory[0])
interpolated_Zn62 = float(interpolated_ratio_theory[1])
df_All['Energy (MeV)'] = interpolated_energy
df_All['Zn62 σ(E)'] = interpolated_Zn62



# Apply function and create a Series of tuples to store
tuples = df_All.apply(lambda row: compute_correction_factor_and_uncertainty(
    row['Isotope'], row['Live Time (seconds)'], isotope_parameters), axis=1)

# Split the tuples into two separate columns in df_All
df_All['Correction Factor'], df_All['Correction Factor Uncertainty'] = zip(*tuples)

df_All['Adjusted Activity Uncertainty (uCi)'] = df_All.apply(
    lambda row: activity_uncertainty(
        row['Adjusted Activity (uCi)'],
        row['Correction Factor'],
        row['Correction Factor Uncertainty'],
        row['Branching Ratio'],
        row['Branching Ratio Uncertainty'],
        row['Net Peak Area'],
        row['Net Area Uncertainty'],
        row['Peak Efficiency'],
        row['Efficiency Uncertainty']
    ), axis=1
)

df_All = add_average_uncertainty(df_All)

df_All['Activity at EOB Uncertainty (uCi)'] = df_All.apply(
    lambda row: calculate_eob_uncertainty(
        row['Half life (s)'],
        row['Activity at EOB (uCi)'],
        row['Lambda Uncertainty'],
        row['Average Adjusted Activity Uncertainty (uCi)'],
        row['Average Adjusted Activity (uCi)'],
        row['Time Difference (seconds)']
    ), axis=1
)

# Compute the energy uncertainty
energy_uncertainty = compute_energy_uncertainty(df_All)
df_All['Energy Uncertainty (MeV)'] = energy_uncertainty

Zn62_EOB = df_All.at[0, 'Activity at EOB (uCi)']
Zn62_cross_section = df_All.at[0, 'Zn62 σ(E)']


df_All['Flux (proton/s)'] = df_All.apply(
    lambda row: compute_flux(
        Zn62_EOB,
        mass_foil_Cu,
        Zn62_cross_section,
        row['Irradiation Time (s)']
    ), axis=1
)

Zn62_EOB_uncertainty = df_All.at[0, 'Activity at EOB Uncertainty (uCi)']
flux = df_All.at[0, 'Flux (proton/s)'] 

df_All['Flux Uncertainty (proton/s)'] = df_All.apply(
    lambda row: compute_flux_uncertainty(
        Zn62_EOB,
        Zn62_EOB_uncertainty,
        flux
    ), axis=1
)

current = flux * 1.6E-19 * 1E6
current_percent_error = (np.abs(current - requested_current) / current) * 100
df_All['Current (uA)'] = current
df_All['Requested Current (uA)'] = requested_current
df_All['Current % Error'] = current_percent_error

df_All['Cross Section (cm^2)'] = df_All.apply(cross_section_for_specific_isotopes, axis=1)
df_All['Cross Section (mb)'] = df_All['Cross Section (cm^2)'] * 1E27

df_All['Cross Section Uncertainty (mb)'] = df_All.apply(
    lambda row: cross_section_uncertainty(
        row['Half life (s)'],
        row['Lambda Uncertainty'],
        row['Time Difference (seconds)'],
        row['Flux (proton/s)'],
        row['Flux Uncertainty (proton/s)'],
        row['Cross Section (mb)']        
    ), axis=1
)

# Apply the styling
styled_df = df_All.style.apply(color_rows_by_isotope, axis=1)

# Now, to export the styled DataFrame to an Excel file
styled_df.to_excel('CrossSectionAnalysisSingleCell_XX_XX_XXXX.xlsx', engine='openpyxl', index=False)
styled_df