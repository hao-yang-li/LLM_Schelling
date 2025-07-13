import os
import time
import base64
import json
import jsonlines
from openai import OpenAI, RateLimitError
from constant import PLATFORM, API_KEY_MAP, BASE_URL_MAP 

class LLM:
    """
    The language model class for generating text using different platforms.

    Args:
        model_name (str): The name of the model.
        platform (PLATFORM): The platform to use.

    Attributes:
        model_name (str): The name of the model.
        platform (PLATFORM): The platform to use.
        client (openai.Client): The OpenAI client.
        completion_tokens (int): The number of tokens used for completion.
        prompt_tokens (int): The number of tokens used for prompt.

    Methods:
        generate: Generate text based on the prompt.
        clear_history: Clear the history of generated text.
        save_history: Save the history of generated text to a file.
    
    """
    def __init__(self, model_name: str, platform: PLATFORM = 'openai', api_key: str = None):
        """
        Initializes an instance of the class.

        Args:
            model_name (str): The name of the model.
            platform (PLATFORM, optional): The platform to use. Defaults to 'openai'.
            api_key (str, optional): The API key for the platform. Defaults to None (use the environment variable).
        """
        self.model_name = model_name
        self.platform = platform
        self.api_key = api_key
        
        self.client = self._init_client(platform)

        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.history = []

        self.batch_list = []

    def _init_client(self, platform: PLATFORM):
        """
        Initialize the OpenAI client with the API key and base URL.

        Args:
            platform (PLATFORM): The platform to use.
        
        Returns:
            openai.Client: The OpenAI client.
        """
        assert platform in API_KEY_MAP, f"Platform {platform} is not supported."
        if self.api_key:
            api_key = self.api_key
        else:
            api_key = os.environ.get(API_KEY_MAP[platform])
        assert api_key, f"API key for platform {platform} is not found."
        base_url = BASE_URL_MAP[platform]
        client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        return client

    def generate(
        self, 
        prompt: list[dict]|str, 
        model:str|None = None, 
        temperature: float = 1.0, 
        max_tokens: int = 4096
    ):
        """
        Generate text based on the prompt.

        Args:
            prompt (str): The prompt for the model.
            model (str|None): The model to use. If None, use the default model.
            temperature (float): The temperature for sampling.
            max_tokens (int): The maximum number of tokens to generate.
        
        Returns:
            str: The generated text.
        """
        if model is None:
            model = self.model_name

        if isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        else:
            messages = prompt

        # save system time when log history
        for m in messages:
            m.update({"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
            self.history.append(m)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                self.completion_tokens += completion.usage.completion_tokens
                self.prompt_tokens += completion.usage.prompt_tokens

                response = completion.choices[0].message.content
                self.history.append({'role': 'assistant', 'content': response, 'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})

                return response
            
            except RateLimitError as e:
                print(f"Rate limit error encountered: {e}. This is attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    # For the first few errors, wait a short time
                    if attempt < 2:
                        wait_time = 5 * (attempt + 1) # Wait 5, 10 seconds
                        print(f"Waiting for {wait_time} seconds before retrying.")
                        time.sleep(wait_time)
                    # If errors persist, wait for a full minute
                    else:
                        print("Waiting for 60 seconds before making a more spaced-out retry.")
                        time.sleep(60)
                else:
                    print("Maximum retries reached. Failing.")
                    raise  # Re-raise the exception after the last attempt
            
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                # For other errors, we might not want to retry, so we fail immediately.
                raise
    
    def get_vision_response(self, image_path: str, prompt: str, model: str = None):
        """
        Generate text based on the image and prompt.

        Args:
            image_path (str): The path to the image.
            prompt (str): The prompt for the model.
            model (str|None): The model to use. If None, use the default model.
        """
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
        # Using the default model if not specified
        if model is None:
            model = self.model_name
            
        # Getting the Base64 string
        base64_image = encode_image(image_path)
        # print(base64_image)

        # Getting the messages
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        self.completion_tokens += completion.usage.completion_tokens
        self.prompt_tokens += completion.usage.prompt_tokens

        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response, 'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})

        return response

    def add_batch_item(
            self,
            custom_id: str, 
            prompt: list[dict]|str,
            image_path: str = None,
            model: str = None,
            end_point: str = '/v1/chat/completions',
            max_tokens: int = 1024,
            temperature: float = 1.0   
    ):
        """
        Add an item to the batch list.

        Args:
            prompt (list[dict]): The prompt for the model.
            model (str|None): The model to use. If None, use the default model.
        """
        assert len(self.batch_list) < 50000, "The batch list is full."

        if model is None:
            model = self.model_name
        else:
            assert model==self.model_name, "The model must be the same as the default model to ensure the batch use the same model."

        system_prompt = []
        user_prompt = []
        image_prompt = []

        if isinstance(prompt, str):
            user_prompt.append(prompt)
        else:
            for p in prompt:
                if p['role'] == 'user':
                    user_prompt.append(
                        {
                            "type": "text",
                            "text": p['content']
                        }
                    )
                else:
                    system_prompt.append(
                        {
                            "type": "text",
                            "text": p['content']
                        }
                    )

        if image_path is not None:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_prompt = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            ]

        messages = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt
                }
            )
        
        if image_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        *image_prompt,
                        *user_prompt
                    ]
                }   
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": user_prompt
                }
            )

        self.batch_list.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": end_point,
                "body": {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            }
        )
        return len(self.batch_list)
    
    def generate_batch(self, batch_list: list[dict] = None, description: str = "nightly eval job"):
        """
        Generate text based on the batch list.

        Args:
            batch_list (list[dict]): The batch list.
        """
        if batch_list is not None:
            self.batch_list = batch_list
        # first save the batch list to a jsonl file
        with jsonlines.open('batch_list.jsonl', 'w') as writer:
            for item in self.batch_list:
                writer.write(item)
        # then use the batch api to upload the batch list
        batch_input_file = self.client.files.create(
            file=open("batch_list.jsonl", "rb"),
            purpose="batch"
        )
        print(batch_input_file)
        # then use the batch api to generate the results
        batch_input_file_id = batch_input_file.id
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": description
            }
        )
        print(batch)
        # save batch as a json file
        with open(f'{batch.id}_metadata.json', 'w') as f:
            json.dump(batch, f)
        return batch
    
    def list_batches(self):
        batches = self.client.batches.list()
        print(batches)
        return batches
    
    def get_batch_result(self, batch_id: str):
        batch = self.client.batches.retrieve(batch_id)
        batch_status = {
            "validating": "the input file is being validated before the batch can begin",
            "failed": "the input file has failed the validation process",
            "in_progress": "the input file was successfully validated and the batch is currently being run",
            "finalizing": "the batch has completed and the results are being prepared",
            "completed": "the batch has been completed and the results are ready",
            "expired": "the batch was not able to be completed within the 24-hour time window",
            "cancelling": "the batch is being cancelled (may take up to 10 minutes)",
            "cancelled": "the batch was cancelled"
        }
        status = batch_status[batch.status]
        print(f"The batch status is {status}")
        if status == "completed" or status == "expired":
            output_file_id = batch.output_file_id
            error_file_id = batch.error_file_id

            output_file_response = self.client.files.content(output_file_id)
            error_file_response = self.client.files.content(error_file_id)

            with open(f'{batch.id}_result.jsonl', 'w') as f:
                f.write(output_file_response.text)
            with open(f'{batch.id}_error.jsonl', 'w') as f:
                f.write(error_file_response.text)

            print(f"The batch result is saved as {batch.id}_result.jsonl")
            print(f"The batch error is saved as {batch.id}_error.jsonl")
        return batch
    
    def cansel_batch(self, batch_id: str):
        batch = self.client.batches.cancel(batch_id)
        print(batch)
        return batch

    def clear_history(self):
        """
        Clear the history of generated text.
        """
        self.history = []

    def save_history(self, path: str):
        """
        Save the history of generated text to a file.

        Args:
            path (str): The path to save the history.
        """
        with jsonlines.open(path, 'w') as writer:
            for item in self.history:
                writer.write(item)

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, platform={self.platform})"



