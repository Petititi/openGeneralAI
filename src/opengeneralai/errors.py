"""Exceptions raised on purpose by opengeneralai.

Their message is meant for the user: the web app shows it as is. Any other exception is a bug,
reported as an internal error without details.
"""


class OpenGeneralAIError(Exception):
    """Base class of the errors the user can act upon."""


class LLMError(OpenGeneralAIError):
    """The call to the LLM failed."""


class LLMAuthError(LLMError):
    """The provider rejected the API key."""


class LLMRateLimitError(LLMError):
    """Too many requests for the provider."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached or did not answer in time."""


class LLMBadRequestError(LLMError):
    """The provider rejected the request (model name, context too long...)."""


class AgentError(OpenGeneralAIError):
    """The agent could not follow its protocol."""


class PlanParseError(AgentError):
    """The LLM answer is not a valid plan or plan decision."""


class ToolCallError(AgentError):
    """The LLM asked for an unknown tool, with invalid arguments, or the tool failed."""

    def __init__(self, message: str, tool_name: str = ""):
        super().__init__(message)
        self.tool_name = tool_name
