def get_pseudo_labels(dataset, model, batch_size, threshold=0.9):

    # Make sure the model is in eval mode.
    model.eval()

    softmax = nn.Softmax(dim=-1)

    # Create a new dataloader for the dataset.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    idxs = []
    labels = []
    print('getting pseudo dataset...')

    # Iterate over the dataset by batches.
    for batch_idx, batch in enumerate(dataloader):
        img, _ = batch

        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(DEVICE))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        for idx, x in enumerate(probs):
            if torch.max(x) > threshold:
                idxs.append(batch_idx*batch_size+idx)
                labels.append(int(torch.argmax(x)))

    print('Finish filtering data. We got {} data. return new pesudo dataset.'.format(len(labels)))

    pesudo_set = pesudoDataset(Subset(dataset, idxs), labels)

    # Turn off the eval mode.
    model.train()
    return pesudo_set