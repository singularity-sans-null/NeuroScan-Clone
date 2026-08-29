from pathlib import Path

def parse_patient_filename(filename):

    stem = Path(filename).stem

    parts = stem.split("_")

    if len(parts) != 3:
        raise ValueError(
            "Filename must follow the format: "
            "PATIENTID_AGE_SEX"
        )

    patient_id = parts[0]

    age = int(parts[1])
    sex = parts[2]
    sex = sex.capitalize()
    sexes = ["Male", "Female"]
    if sex in sexes:
        sex = sex
    elif f"{sex[0]+sexes[0].lstrip("M")}" in sexes:
        sex = sexes[0]
    elif f"{sex[0]+sexes[1].lstrip("MF")}" in sexes:
        sex = sexes[1]
    else:
        raise ValueError(
            "Sex must be Male or Female"
    )
    print(f"{patient_id}\n {age}\n {sex}\n")
    return patient_id, age, sex

parse_patient_filename("N5001_24_F.png")