def generate_global_features(mol: RDKitMol,
                             features_generators: List[str]) -> np.ndarray:
    """Helper function for generating global features for a RDKit mol based on the given list of feature generators to be used.

    Parameters
    ----------
    mol: RDKitMol
        RDKit molecule to be featurized
    features_generators: List[str]
        List of names of the feature generators to be used featurization

    Returns
    -------
    global_features_array: np.ndarray
        Array of global features

    Examples
    --------
    >>> from rdkit import Chem
    >>> import deepchem as dc
    >>> mol = Chem.MolFromSmiles('C')
    >>> features_generators = ['morgan']
    >>> global_features = dc.feat.molecule_featurizers.dmpnn_featurizer.generate_global_features(mol, features_generators)
    >>> type(global_features)
    <class 'numpy.ndarray'>
    >>> len(global_features)
    2048
    >>> nonzero_features_indices = global_features.nonzero()[0]
    >>> nonzero_features_indices
    array([1264])
    >>> global_features[nonzero_features_indices[0]]
    1.0

    """
    global_features: List[np.ndarray] = []
    available_generators = GraphConvConstants.FEATURE_GENERATORS

    for generator in features_generators:
        if generator in available_generators:
            global_featurizer = available_generators[generator]
            if mol.GetNumHeavyAtoms() > 0:
                global_features.extend(global_featurizer.featurize(mol)[0])
            # for H2
            elif mol.GetNumHeavyAtoms() == 0:
                # not all features are equally long, so used methane as dummy molecule to determine length
                global_features.extend(
                    np.zeros(
                        len(
                            global_featurizer.featurize(
                                Chem.MolFromSmiles('C'))[0])))
        else:
            logger.warning(f"{generator} generator is not available in DMPNN")

    global_features_array: np.ndarray = np.asarray(global_features)

    # Fix nans in features
    replace_token = 0
    global_features_array = np.where(np.isnan(global_features_array),
                                     replace_token, global_features_array)

    return global_features_array