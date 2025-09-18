# backend/async_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncMLProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def process_large_dataset(self, df, config):
        """Asynchroniczne przetwarzanie w chunkach"""
        chunks = np.array_split(df, 4)
        
        tasks = [
            asyncio.create_task(self._process_chunk(chunk, config))
            for chunk in chunks
        ]
        
        results = await asyncio.gather(*tasks)
        return self._merge_results(results)
    
    async def _process_chunk(self, chunk, config):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._train_model, 
            chunk, 
            config
        )