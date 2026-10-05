
import litellm

def get_providers_and_models():
    providers = []
    provider_to_models = {}
    mbp = litellm.models_by_provider
    if isinstance(mbp, dict) and mbp:
        provider_to_models = {str(k): sorted(set(v)) for k, v in mbp.items() if v}
    else:
        # Fallback: derive mapping from litellm.model_list using get_llm_provider
        ml = litellm.model_list
        tmp = {}
        for m in ml:
            try:
                _m, prov, *_ = litellm.get_llm_provider(m)
                if prov:
                    tmp.setdefault(str(prov), []).append(m)
            except Exception:
                continue
        provider_to_models = {k: sorted(set(v)) for k, v in tmp.items()}
    providers = sorted(provider_to_models.keys())
    return providers, provider_to_models
