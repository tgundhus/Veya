namespace Veya;

/// <summary>
/// Anything that can answer a <see cref="QuestionSet"/> about a state in one call: a single
/// <see cref="VeyaEngine"/>, or a <see cref="VeyaRouter"/> that picks one first. Mirrors Python's
/// duck-typed <c>predict</c>/<c>system_one</c> (<c>shortlist._call_predict</c>), so
/// <see cref="VeyaShortlist.Predict"/> works over either.
/// </summary>
public interface IVeyaPredictor
{
    /// <summary>Answer every question in <paramref name="questions"/> about <paramref name="state"/>.</summary>
    VeyaResult Predict(object? state, QuestionSet questions);
}
