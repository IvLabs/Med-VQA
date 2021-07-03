import config.Config as config
import model
import trainer
import _utils



answers_dict = _utils.Datamake.make_ans_dict(config.TRAIN_DATA_DICT_PATH)

train_loader, pad_idx, vocab = _utils.Datamake.get_loader(image_path=config.TRAIN_IMG_PATH, vqa_path=config.TRAIN_DATA_DICT_PATH)
valid_loader, val_pad_idx, val_vocab = _utils.Datamake.get_loader_val(vocab=vocab, image_path=config.TEST_IMG_PATH, vqa_path=config.TEST_DATA_DICT_PATH)


optimizer = torch.optim.Adam(stud.parameters(),lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,verbose=True)

num_classes = len(answers_dict)
vocab_len = len(vocab[0])
coattention = CoAttention(num_embeddings=vocab_len, num_classes=num_classes, embed_dim=512, k=64).to(config.DEVICE)    
visualfeatures = VisualFeatures(config.DEVICE=config.DEVICE).to(config.DEVICE)
stud = Identity(coattention, visualfeatures, vocab_len)
stud = stud.to(config.DEVICE)

trainer.Trainer.train(stud, train_loader, valid_loader, answers_dict, optimizer, criterion, scheduler, config)