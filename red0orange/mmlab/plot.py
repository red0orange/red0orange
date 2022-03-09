import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

sns.set_style('darkgrid')


def custom_plot_tensorboard_data(tf_event_path):
    """专门用于目前mmclassification的tensorboard数据绘图
    """
    # laod data
    ea = event_accumulator.EventAccumulator(tf_event_path) 
    ea.Reload()
    # process data
    train_loss = [[i.step,i.value] for i in ea.scalars.Items('interval_after_train_iter/loss')]
    val_loss = [[i.step,i.value] for i in ea.scalars.Items('statistics_after_val_epoch/loss')]
    val_acc = [[i.step,i.value] for i in ea.scalars.Items('direct_after_train_epoch/accuracy_top-1')]
    # create fig
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    loss_axe, acc_axe = axes
    # plot fig
    df = pd.DataFrame(dict(step=[i[0] for i in train_loss], value=[i[1] for i in train_loss]))
    sns.lineplot(x="step", y="value", data=df, ax=loss_axe)
    df = pd.DataFrame(dict(step=[i[0] for i in val_loss], value=[i[1] for i in val_loss]))
    sns.lineplot(x="step", y="value", data=df, ax=loss_axe)
    loss_axe.legend(["Train Loss", "Valid Loss"])
    # loss_axe.set_ylim([40, 90])
    loss_axe.set_title("Loss")
    
    df = pd.DataFrame(dict(step=[i[0] for i in val_acc], value=[i[1] for i in val_acc]))
    sns.lineplot(x="step", y="value", data=df, marker='o', dashes=False, ax=acc_axe)
    acc_axe.legend(["Valid Acc"], loc="lower right")
    acc_axe.set_ylim([40, 90])
    acc_axe.set_title("Valid Acc")
    max_acc = max(df['value'])
    for x, y in zip(df['step'], df['value']):
        if y == max_acc:
            acc_axe.text(x = x,  # x-coordinate position of data label
                         y = y + 1.2,  # y-coordinate position of data label, adjusted to be 150 below the data point
                         s = '{:.2f}%'.format(y),  # data label, formatted to ignore decimals
                         color = 'blue')  # set colour of line
    return fig


def custom_save_tensorboard_plot(tf_logs, save_paths):
    if not isinstance(tf_logs, list):
        tf_logs = [tf_logs]
    if not isinstance(save_paths, list):
        save_paths = [save_paths]
    assert len(tf_logs) == len(save_paths)
    
    for tf_log, save_path in zip(tf_logs, save_paths):
        fig = custom_plot_tensorboard_data(tf_log)
        fig.savefig(save_path)
    pass


if __name__ == "__main__":
    plot_data = {
        "Resnet34.png": "",
        "Resnet34.png": "",
        "Resnet34.png": "",
        "Resnet34.png": "",
        "Resnet34.png": "",
    }
    custom_save_tensorboard_plot("events.out.tfevents.1639190431.2bfa6bbc6b69.796165.0", "test.png")
    pass