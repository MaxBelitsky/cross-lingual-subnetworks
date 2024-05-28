import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from cross_lingual_subnets.cka import cka


BASE_OUTPUT_PATH = "outputs/images/"
FIGSIZE = (10, 7)
LANGUAGES = ["en", "es", "ru", "ar", "de", "hi", "zh"]
MODEL_TYPES = ["base", "finetuned", "sub"]
MODEL_TYPE_TO_EXPERIMENT = {
    "base": "Experiments.XLMR_BASE",
    "finetuned": "Experiments.XLMR_MLM_FINETUNED",
    "sub": {
        "en": "Experiments.EN_SUB_MLM_FINETUNED",
        "es": "Experiments.ES_SUB_MLM_FINETUNED",
        "de": "Experiments.DE_SUB_MLM_FINETUNED",
        "ru": "Experiments.RU_SUB_MLM_FINETUNED",
        "zh": "Experiments.ZH_SUB_MLM_FINETUNED",
        "hi": "Experiments.HI_SUB_MLM_FINETUNED",
        "ar": "Experiments.AR_SUB_MLM_FINETUNED",
    },
}


def load_encodings(
    path_to_encodings: str,
    max_length: int = None,
    languages: list = LANGUAGES,
    model_types: list = MODEL_TYPES,
    model_type_to_experiment: dict = MODEL_TYPE_TO_EXPERIMENT,
) -> dict:
    """Loads sentence representations.

    :param path_to_encodings: base path to the directory where all the encodings are
    stored e.g. encodings/
    :param max_length: limit to cut representations to save computation time, defaults
    to None
    :param languages: languages to load encodings for, defaults to ["en", "es", "ru",
     "ar", "de", "hi", "zh"]
    :param model_types: model types to load encodings for, defaults to ["base",
     "finetuned", "sub"]
    :param model_type_to_experiment: dictionary for each language which sub-directory
     was used, defaults to MODEL_TYPE_TO_EXPERIMENT
    :return: loaded dictionary
    """
    full_sub = {}
    for lang in languages:
        full_sub[lang] = {}
        for model_type in model_types:
            experiment = (
                model_type_to_experiment[model_type][lang]
                if model_type == "sub"
                else model_type_to_experiment[model_type]
            )
            exp_lang = os.path.join(experiment, f"{lang}.pt")
            full_path = os.path.join(path_to_encodings, exp_lang)

            encoding_dict = torch.load(full_path)
            if max_length is not None and max_length > 0:
                new_encoding_dict = {}
                for layer, values in encoding_dict.items():
                    new_encoding_dict[layer] = values[:max_length]
                encoding_dict = new_encoding_dict

            full_sub[lang][model_type] = encoding_dict

    return full_sub


def save_img(savename: str) -> None:
    """Save image using savename.

    :param savename: name of the image.
    """
    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    if savename is not None:
        plt.savefig(os.path.join(BASE_OUTPUT_PATH, savename))


def cka_cross_layer(
    repr1: torch.Tensor,
    repr2: torch.Tensor,
    xlabel: str,
    ylabel: str,
    title: str = None,
    savename: str = None,
) -> None:
    """Plot heatmap of cross-layer CKA similarities of representations.

    :param repr1: (num_inputs x hid_dim) size tensor of representations
    :param repr2: (num_inputs x hid_dim) size tensor of representations
    :param xlabel: Name of the first encoder
    :param ylabel: Name of the second encoder
    :param title: name the plot, defaults to None
    :param savename: where to save the image, defaults to None
    """
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


def cka_cross_layer_all_languages(
    full_sub: dict,
    xlabel: str,
    ylabel: str,
    exp_name1: str,
    exp_name2: str,
    savename: str = None,
    figsize: tuple = (7, 13),
):
    """_summary_

    :param full_sub: _description_
    :param xlabel: _description_
    :param ylabel: _description_
    :param exp_name1:
    :param exp_name2: _description_
    :param savename: how to save the image, defaults to None
    :param figsize: size of the figure, defaults to (7, 13)
    """
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

    # Delete the last (if unused) subplot
    if len(languages) % 2 != 0:
        fig.delaxes(axs[-1])

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    save_img(savename)


def plot_per_layer_lineplot(
    res_df: pd.DataFrame,
    title: str = None,
    legend: bool = True,
    figsize: tuple = FIGSIZE,
    linewidth: int = 3,
    dashes: bool = False,
    marker: str = "o",
    markersize: int = 8,
) -> None:
    sns.lineplot(
        data=res_df,
        marker=marker,
        legend=legend,
        linewidth=linewidth,
        dashes=dashes,
        markersize=markersize,
    )
    plt.grid()
    num_layers = len(res_df[res_df.columns[0]])
    plt.xticks(range(num_layers))
    plt.title(title)
    plt.xlabel("Layers")
    plt.ylabel("CKA Similarity")


def cka_layer_by_layer(
    full_sub: dict,
    exp1: str,
    exp2: str,
    source: str | None = None,
    savename: str = None,
    figsize: tuple = FIGSIZE,
    title: str = None,
    legend: bool = True,
    linewidth: int = 3,
    dashes: bool = False,
    marker: str = "o",
    markersize: int = 8,
) -> pd.DataFrame:
    cka_results = dict()
    plt.figure(figsize=figsize)
    for lang, vals in full_sub.items():
        # Compute similarity for some source language
        # or for the same language but between different models
        repr1 = vals[exp1] if source is None else full_sub[source[exp1]]
        repr2 = vals[exp2]

        res = []
        for layer_id in range(len(repr1)):
            layer_repr1 = repr1[layer_id].detach()
            layer_repr2 = repr2[layer_id].detach()
            res.append(cka(layer_repr1, layer_repr2))
        cka_results[f"{lang}_{exp1}-{lang}_{exp2}"] = res

    df = pd.DataFrame(cka_results)
    plot_per_layer_lineplot(
        res_df=df,
        title=title,
        legend=legend,
        linewidth=linewidth,
        dashes=dashes,
        marker=marker,
        markersize=markersize,
    )
    save_img(savename)
    return df


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
