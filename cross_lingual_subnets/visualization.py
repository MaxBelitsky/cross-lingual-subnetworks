import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cross_lingual_subnets.cka import cka

BASE_OUTPUT_PATH = "outputs/images/"
FIGSIZE = (10, 7)


def save_img(savename: str) -> None:
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    if savename is not None:
        plt.savefig(os.path.join(BASE_OUTPUT_PATH, savename))


def cka_cross_layer(
    repr1, repr2, xlabel: str, ylabel: str, title: str = None, savename: str = None
) -> None:
    cka_results = dict()
    for i in range(len(repr1)):
        layer_reprs1 = repr1[i].detach()
        res = []
        for j in range(len(repr2)):
            layer_reprs2 = repr2[j].detach()
            res.append(cka(layer_reprs1, layer_reprs2))
        cka_results[i] = res

    df = pd.DataFrame(cka_results)
    df = df.sort_index(ascending=False)

    ax = sns.heatmap(df)
    ax.set(xlabel=f"{xlabel} layer", ylabel=f"{ylabel} layer")
    if title is not None:
        ax.set(title=title)
    if savename is not None:
        plt.savefig(os.path.join(BASE_OUTPUT_PATH, savename))


def cka_layer_by_layer(
    full_sub: dict,
    exp1: str,
    exp2: str,
    savename: str = None,
    title: str = None,
    legend: bool = True,
    figsize: tuple = FIGSIZE,
    linewidth: int = 3,
    dashes: bool = False,
    marker: str = "o",
    markersize: int = 8,
) -> None:
    cka_results = dict()
    plt.figure(figsize=figsize)
    for lang, vals in full_sub.items():
        full = vals[exp1]
        sub = vals[exp2]

        res = []
        for layer_id in range(len(full)):
            res.append(cka(full[layer_id].detach(), sub[layer_id].detach()))
        cka_results[f"{lang}_{exp1}-{lang}_{exp2}"] = res

    df = pd.DataFrame(cka_results)

    sns.lineplot(
        data=df,
        marker=marker,
        legend=legend,
        linewidth=linewidth,
        dashes=dashes,
        markersize=markersize,
    )
    plt.grid()
    plt.xticks(range(12))
    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("CKA Similarity")

    save_img(savename)


def cka_layer_by_layer_langs(
    full_sub: dict,
    exp_name1: str,
    exp_name2: str,
    source: str = "en",
    savename: str = None,
    title: str = None,
    figsize: tuple = FIGSIZE,
) -> pd.DataFrame:
    plt.figure(figsize)
    cka_results = dict()
    source_vals = full_sub[source][exp_name1]
    for lang, vals in full_sub.items():
        if lang == source:
            continue

        ref_vals = vals[exp_name2]

        res = []
        for layer_id in range(len(ref_vals)):
            res.append(cka(source_vals[layer_id].detach(), ref_vals[layer_id].detach()))
        cka_results[f"{source}_{exp_name1}-{lang}_{exp_name2}"] = res

    df = pd.DataFrame(cka_results)

    sns.lineplot(data=df, markers=True)
    plt.grid()
    plt.xticks(range(12))
    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("CKA Similarity")

    save_img(savename)

    return df


def cka_cross_layer_all_languages(
    full_sub: dict,
    xlabel: str,
    ylabel: str,
    exp_name1: str,
    exp_name2: str,
    savename: str = None,
    figsize: tuple = (7, 13),
):
    languages = full_sub.keys()
    fig, axs = plt.subplots(round(len(languages) / 2), 2, figsize=figsize)
    axs = axs.reshape(-1)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    for i, (lang, preds) in enumerate(full_sub.items()):
        cka_results = dict()

        for k in range(len(preds[exp_name1])):
            layer_reprs1 = preds[exp_name1][k].detach()
            res = []
            for j in range(len(preds[exp_name2])):
                layer_reprs2 = preds[exp_name2][j].detach()
                res.append(cka(layer_reprs1, layer_reprs2))
            cka_results[k] = res

        df = pd.DataFrame(cka_results)
        df = df.sort_index(ascending=False)

        ax = sns.heatmap(df, ax=axs[i], cbar=i == 0, cbar_ax=None if i else cbar_ax)
        ax.set(title=lang)

    # Delete the last unused subplot
    if len(languages) % 2 != 0:
        fig.delaxes(axs[-1])

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    save_img(savename)


def cka_diff_barplots(df, savename: str = None, figsize: tuple = FIGSIZE):
    lang_pairs = df.index
    fig, axs = plt.subplots(len(lang_pairs), 1, figsize=figsize)
    ymin = df.min().min()
    ymax = df.max().max()
    cols = sns.color_palette()
    for i, lang_pair in enumerate(lang_pairs):
        sns.barplot(data=df.loc[lang_pair], ax=axs[i], color=cols[i])
        axs[i].set_ylim([ymin, ymax])
        axs[i].grid()

    save_img(savename)
