import typing as tp


def prepare_input_per_users(
    prompt: str,
    user_profile: str,
    prev_interactions: tp.Dict[str, str],
    top_k: int
) -> str:
    
    return prompt.format(user_profile=user_profile, item_ids_with_meta=prev_interactions, top_k=top_k)