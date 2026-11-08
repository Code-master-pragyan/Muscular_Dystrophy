    manifest_path: str | Path,
    split_path: str | Path,
) -> Dict[str, pd.DataFrame]:
    """Load the existing manifest/split and prepare model-ready metadata.

    Parameters
    ----------
    manifest_path:
        Path to Dataset_B_Image_Mask_Manifest.csv.
    split_path:
        Path to Dataset_B_Subject_Level_Split.csv.

    Returns
    -------
    dict
        DataFrames keyed by ``train``, ``validation`` and ``test``.
        Each row remains an image record. ``label`` is the zero-based
        class index (grade 1 -> 0, ..., grade 4 -> 3), and ``group_id``
        identifies a subject-muscle-side group.
    """
    manifest_path = Path(manifest_path)
    split_path = Path(split_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Subject split file not found: {split_path}")

    manifest_df = pd.read_csv(manifest_path)
    split_df = pd.read_csv(split_path)

    missing_manifest = REQUIRED_MANIFEST_COLUMNS.difference(manifest_df.columns)
    missing_split = REQUIRED_SPLIT_COLUMNS.difference(split_df.columns)

    if missing_manifest:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(sorted(missing_manifest))
        )
    if missing_split:
        raise ValueError(
            "Subject split is missing required columns: "
            + ", ".join(sorted(missing_split))
        )

    if split_df["subject_id"].duplicated().any():
        duplicates = split_df.loc[
            split_df["subject_id"].duplicated(keep=False), "subject_id"
        ].unique()
        raise ValueError(
            "Subject split must contain one row per subject. "
            f"Duplicate subject IDs include: {duplicates[:10].tolist()}"
        )

    unexpected_partitions = (
        set(split_df["partition"].dropna().astype(str)) - set(VALID_PARTITIONS)
    )
    if unexpected_partitions:
        raise ValueError(
            f"Unexpected partition values: {sorted(unexpected_partitions)}"
        )

    # Retain only records with an available Heckmatt grade, as in the notebook.
    labelled_df = manifest_df[
        manifest_df["heckmatt_grade"].notna()
    ].copy()

    # Attach the previously approved subject-level split; do not resplit.
    labelled_df = labelled_df.merge(
        split_df[["subject_id", "partition"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )

    if labelled_df["partition"].isna().any():
        missing_subjects = (
            labelled_df.loc[labelled_df["partition"].isna(), "subject_id"]
            .dropna()
            .unique()
        )
        raise ValueError(
            "Some labelled records have no approved subject partition. "
            f"Example subject IDs: {missing_subjects[:10].tolist()}"
        )

    grades = pd.to_numeric(labelled_df["heckmatt_grade"], errors="coerce")
    if grades.isna().any() or not grades.isin(VALID_GRADES).all():
        invalid = sorted(grades[~grades.isin(VALID_GRADES)].dropna().unique().tolist())
        raise ValueError(
            "Heckmatt grades must be integers in {1, 2, 3, 4}. "
            f"Invalid values: {invalid}"
        )

    labelled_df["heckmatt_grade"] = grades.astype(int)
    labelled_df["label"] = labelled_df["heckmatt_grade"] - 1

    # Preserve the group definition used for group-level evaluation.
    labelled_df["group_id"] = (
        labelled_df["subject_id"].astype(str)
        + "_"
        + labelled_df["muscle_code"].astype(str)
        + "_"
        + labelled_df["side_code"].astype(str)
    )

    partitions = {
        name: labelled_df[labelled_df["partition"] == name].copy().reset_index(drop=True)
        for name in VALID_PARTITIONS
    }

    # Ensure that no participant occurs in more than one partition.
    subject_sets = {
        name: set(frame["subject_id"].dropna().unique())
        for name, frame in partitions.items()
    }
    for i, first in enumerate(VALID_PARTITIONS):
        for second in VALID_PARTITIONS[i + 1 :]:
            overlap = subject_sets[first].intersection(subject_sets[second])
            if overlap:
                raise ValueError(
                    f"Subject leakage between {first} and {second}: "
                    f"{sorted(overlap)[:10]}"
                )

    if sum(len(frame) for frame in partitions.values()) != len(labelled_df):
        raise RuntimeError("Partitioning lost or duplicated labelled image records.")
