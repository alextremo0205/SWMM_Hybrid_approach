# From https://www.inside-machinelearning.com/en/the-ideal-pytorch-function-to-train-your-model-easily/

import time

def train(model, optimizer, scheduler, loss_fn, train_dl, val_dl, epochs=100, device='cpu'):

    print_initial_message(model, optimizer, epochs, device)

    history = initialize_history()

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):

        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        model.train()
        train_loss         = 0.0

        for batch in train_dl:

            optimizer.zero_grad()

            x    = batch.to(device)            
            y    = batch.y[:,0].reshape(-1,1).to(device)
            
            yhat = model(x)
            
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
        
        scheduler.step()
        train_loss  = train_loss / len(train_dl.dataset)

        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        val_loss       = 0.0

        for batch in val_dl:

            x    = batch.to(device)
            y    = batch.y[:,0].reshape(-1,1).to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            val_loss         += loss.data.item() * x.size(0)

        val_loss = val_loss / len(val_dl.dataset)

        printCurrentStatus(epochs, epoch, train_loss, val_loss)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    # END OF TRAINING LOOP


    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history

def printCurrentStatus(epochs, epoch, train_loss, val_loss):
    if epoch == 1 or epoch % 10 == 0:
      print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
                (epoch, epochs, train_loss, val_loss))


def initialize_history():
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    return history


def print_initial_message(model, optimizer, epochs, device):
    print('train() called:model=%s, opt=%s(lr=%f), epochs=%d,device=%s\n' %
                        
                        (type(model).__name__,              \
                        type(optimizer).__name__,           \
                        optimizer.param_groups[0]['lr'],    \
                        epochs, device)
            )