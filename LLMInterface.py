"""Interface for interacting with LLM providers."""


class LLMInterface:
    def __init__(self, provider: str, api_key: str, model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openai":
            # Updated initialization for newer OpenAI client
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai
        elif provider == "deepseek":
            # Add appropriate initialization for deepseek
            import deepseek
            self.client = deepseek.DeepseekAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        if self.provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Updated for newer client, using the specified model
            response = self.client.chat.completions.create(
                model=self.model,  # Use the model specified in initialization
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            model = self.client.GenerativeModel('gemini-pro')
            if system_prompt:
                response = model.generate_content(
                    [system_prompt, prompt],
                    generation_config={"temperature": temperature}
                )
            else:
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )
            return response.text

        elif self.provider == "deepseek":
            # Implement deepseek API call here based on their documentation
            # This is a placeholder - you'll need to adjust based on actual deepseek API
            response = self.client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature
            )
            return response.text