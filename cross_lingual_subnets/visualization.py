from cross_lingual_subnets.cka import cka
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cross_lingual_subnets.cca_core import get_cca_similarity


def cka_cross_layer(
    repr1, repr2, xlabel: str, ylabel: str, title: str = None, savename: str = None
):
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
    ax.set(xlabel=f"{xlabel} Layer", ylabel=f"{ylabel} Layer")
    if title is not None:
        ax.set(title=title)
    if savename is not None:
        plt.savefig(f"images/{savename}")


def cka_layer_by_layer(full_sub: dict, savename: str = None, title: str = None):
    cka_results = dict()
    for lang, vals in full_sub.items():
        full = vals["full"]
        sub = vals["sub"]

        res = []
        for layer_id in range(len(full)):
            res.append(cka(full[layer_id].detach(), sub[layer_id].detach()))
        cka_results[f"{lang}_full-{lang}_sub"] = res

    df = pd.DataFrame(cka_results)

    sns.lineplot(data=df, markers=True)
    plt.grid()
    plt.xticks(range(12))
    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("CKA Similarity")
    if savename is not None:
        plt.savefig(f"images/{savename}")


def cka_layer_by_layer_langs(
    full_sub: dict,
    exp_name1: str,
    exp_name2: str,
    source: str = "en",
    savename: str = None,
    title: str = None,
):
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
    if savename is not None:
        plt.savefig(f"images/{savename}")

    return df


def cka_cross_layer_all_languages(
    full_sub: dict,
    xlabel: str,
    ylabel: str,
    exp_name1: str,
    exp_name2: str,
    savename: str = None,
):
    languages = full_sub.keys()
    fig, axs = plt.subplots(round(len(languages) / 2), 2, figsize=(7, 13))
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

    if savename is not None:
        plt.savefig(f"images/{savename}")


def cka_diff_barplots(df, savename: str = None):
    lang_pairs = df.index
    fig, axs = plt.subplots(len(lang_pairs), 1, figsize=(7, 10.5))
    ymin = df.min().min()
    ymax = df.max().max()
    cols = sns.color_palette()
    for i, lang_pair in enumerate(lang_pairs):
        sns.barplot(data=df.loc[lang_pair], ax=axs[i], color=cols[i])
        axs[i].set_ylim([ymin, ymax])
        axs[i].grid()
    if savename is not None:
        plt.savefig(f"images/{savename}")


def svcca_cross_layer(
    repr1, repr2, xlabel: str = None, ylabel: str = None, savename: str = None
):
    cka_results = dict()
    for i in range(len(repr1)):
        # Input must be number of neurons (encoding dim?) by datapoints
        layer_reprs1 = repr1[i].detach().T
        res = []
        for j in range(len(repr2)):
            layer_reprs2 = repr2[j].detach().T
            score = get_cca_similarity(layer_reprs1, layer_reprs2, verbose=False)[
                "mean"
            ][0]
            res.append(score)
        cka_results[i] = res

    df = pd.DataFrame(cka_results)
    df = df.sort_index(ascending=False)

    ax = sns.heatmap(df)
    if xlabel is not None and ylabel is not None:
        ax.set(xlabel=f"{xlabel} layer", ylabel=f"{ylabel} layer")
    if savename is not None:
        plt.savefig(f"images/{savename}")
