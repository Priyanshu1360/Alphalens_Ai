import re
import logging
from guardrails.validators import Validator, register_validator, ValidationResult, PassResult, FailResult
from guardrails import Guard

LOGGER = logging.getLogger("guardrails")

# Common injection vectors
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard (all )?previous instructions",
    r"system prompt",
    r"you are now (a|an|act as)",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# Simple PII: SSN, Emails, and Phones
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
PHONE_REGEX = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")


@register_validator(name="custom/prompt_injection", data_type="string")
class PromptInjectionValidator(Validator):
    """Validator to block prompt injection attacks."""
    def validate(self, value, metadata=None) -> ValidationResult:
        if INJECTION_REGEX.search(value):
            LOGGER.warning(f"Blocked potential prompt injection: {value}")
            # We return a FailResult, which will trigger the on_fail behavior of the Guard
            return FailResult(error_message="Security Guardrail: Your query contains unauthorized instruction patterns.")
        return PassResult()


@register_validator(name="custom/pii_masking", data_type="string")
class PIIMaskingValidator(Validator):
    """Validator to mask PII from input."""
    def validate(self, value, metadata=None) -> ValidationResult:
        sanitized = value
        masked = False
        
        if SSN_REGEX.search(sanitized):
            LOGGER.info("Masked SSN in query.")
            sanitized = SSN_REGEX.sub("[SSN_REDACTED]", sanitized)
            masked = True
            
        if EMAIL_REGEX.search(sanitized):
            LOGGER.info("Masked Email in query.")
            sanitized = EMAIL_REGEX.sub("[EMAIL_REDACTED]", sanitized)
            masked = True
            
        if PHONE_REGEX.search(sanitized):
            LOGGER.info("Masked Phone Number in query.")
            sanitized = PHONE_REGEX.sub("[PHONE_REDACTED]", sanitized)
            masked = True
            
        if masked:
            # We return FailResult with fix_value so the Guard can "fix" the string
            return FailResult(
                error_message="Contains PII",
                fix_value=sanitized
            )
        return PassResult()


# Initialize the Guard with our lightweight custom validators
input_guard = Guard().use(
    PromptInjectionValidator(on_fail="exception")
).use(
    PIIMaskingValidator(on_fail="fix")
)


def apply_input_guardrails(query: str) -> str:
    """
    Applies Layer 1 input governance using Guardrails AI:
    1. Blocks prompt injection attacks (Raises Exception on fail).
    2. Masks sensitive PII (SSN, Email) before processing (Fixes on fail).
    """
    if not query:
        return query
        
    try:
        # Validate the input string using the Guard
        result = input_guard.validate(query)
        
        # If the PII validator fired, it will "fix" the string and store it in validated_output
        # If it passed without fixes, validated_output is the original string.
        # However, if validated_output is None (e.g. it failed but wasn't fixed), we fallback.
        if result.validated_output is not None:
            return result.validated_output
        return query
    except Exception as e:
        # Guardrails will raise an Exception (typically Exception or ValidationError) 
        # when the PromptInjectionValidator fails because of on_fail="exception".
        if type(e).__name__ != 'ValidationError' and "unauthorized instruction" not in str(e):
            raise e # Don't mask generic bugs like NameError
        
        LOGGER.warning(f"Guardrails exception: {e}")
        raise ValueError("Security Guardrail: Your query contains unauthorized instruction patterns.") from e
