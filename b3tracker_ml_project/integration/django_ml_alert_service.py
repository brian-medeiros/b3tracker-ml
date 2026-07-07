from __future__ import annotations

from pathlib import Path

import pandas as pd


def carregar_modelo(path_modelo: str | Path):
    """Carrega o modelo treinado pelo pipeline.

    Este arquivo e um exemplo de integracao. No B3Tracker real, essa funcao
    poderia ser chamada por uma task agendada antes do envio de e-mails.
    """
    path_modelo = Path(path_modelo)
    if path_modelo.suffix == ".pkl":
        import pickle

        with path_modelo.open("rb") as file:
            return pickle.load(file)

    import joblib

    return joblib.load(path_modelo)


def gerar_alertas_ml(model_bundle: dict, dataset_recente: pd.DataFrame) -> pd.DataFrame:
    """Gera sinais COMPRA/VENDA/NEUTRO para integrar ao alerta por e-mail."""
    from b3tracker_ml.inference import predict_latest_alerts

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    config = model_bundle["config"]
    alerts = predict_latest_alerts(
        model=model,
        dataset=dataset_recente,
        feature_columns=feature_columns,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
        min_margin=config.min_margin,
    )
    return pd.DataFrame([alert.__dict__ for alert in alerts])


def exemplo_texto_email(alerta: pd.Series) -> str:
    """Monta um texto simples que poderia ser anexado ao e-mail atual do B3Tracker."""
    return (
        f"Ativo: {alerta['ticker']}\n"
        f"Data de referencia: {alerta['date']}\n"
        f"Sinal ML: {alerta['decision']}\n"
        f"Probabilidade NEUTRO: {alerta['probability_neutral']:.1%}\n"
        f"Probabilidade COMPRA: {alerta['probability_buy']:.1%}\n"
        f"Probabilidade VENDA: {alerta['probability_sell']:.1%}\n\n"
        "Observacao: este sinal e um apoio quantitativo e nao representa recomendacao automatica."
    )
