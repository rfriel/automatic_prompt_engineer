from automatic_prompt_engineer.flan import FlanForward, FLAN_NAME


FLAN_APE = FlanForward.load(FLAN_NAME, bs=64)