import dataset

if __name__ == '__main__':
    # Get the pokemon dataloader
    poke_dataloader = dataset.get_data_loader(batch_size=1, shuffle=True, num_workers=1)
