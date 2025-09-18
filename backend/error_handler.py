# backend/error_handler.py
import logging
from functools import wraps

def handle_ml_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MemoryError:
            st.error("💾 Za mało pamięci - spróbuj mniejszy dataset")
            logging.error(f"Memory error in {func.__name__}")
        except TimeoutError:
            st.error("⏱️ Timeout - operacja trwała za długo")  
            logging.error(f"Timeout in {func.__name__}")
        except Exception as e:
            st.error(f"❌ Nieoczekiwany błąd: {str(e)}")
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None
    return wrapper