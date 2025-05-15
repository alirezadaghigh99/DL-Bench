def random_text_classification_dataset(num_samples=10, max_length=60, num_classes=2, multi_label=False, vocab_size=10,
                                       device='cpu', target_labels='inferred', dtype=torch.long):

    if target_labels not in ['explicit', 'inferred']:
        raise ValueError(f'Invalid test parameter value for target_labels: {str(target_labels)}')

    if num_classes > num_samples:
        raise ValueError('When num_classes is greater than num_samples the occurrence of each '
                         'class cannot be guaranteed')

    if multi_label:
        data = [(
                    torch.randint(vocab_size, (max_length,), dtype=dtype, device=device),
                    np.sort(random_labeling(num_classes, multi_label)).tolist()
                 )
                for _ in range(num_samples)]
    else:
        data = [
            (torch.randint(10, (max_length,), dtype=dtype, device=device),
             random_labeling(num_classes, multi_label))
            for _ in range(num_samples)]

    data = assure_all_labels_occur(data, num_classes, multi_label=multi_label)

    target_labels = None if target_labels == 'inferred' else np.arange(num_classes)
    return PytorchTextClassificationDataset(data, multi_label=multi_label, target_labels=target_labels)