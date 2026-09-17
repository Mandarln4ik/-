package dev.neuroforge.core

/**
 * Evaluates an arithmetic expression.
 *
 * Exists because arithmetic is the thing small models are most confidently wrong about,
 * and it is also the one kind of question that has a single right answer computable in a
 * few lines. A recursive-descent parser rather than anything clever: the grammar is four
 * levels deep, and pulling in an expression library for it would be more code to audit
 * than this whole file.
 *
 * Deliberately *not* a general expression language. There is no variable lookup, no
 * function call and no exponent-by-loop, so there is nothing here that can be made to
 * consume unbounded time or memory by a model that has been talked into passing something
 * hostile — which matters, because the string comes from the model rather than the user.
 */
fun evaluateArithmetic(expression: String): Double {
  val parser = ArithmeticParser(expression)
  val value = parser.expression()
  parser.expectEnd()
  return value
}

private class ArithmeticParser(private val text: String) {
  private var at = 0

  /** expression := term (('+' | '-') term)* */
  fun expression(): Double {
    var value = term()
    while (true) {
      when {
        eat('+') -> value += term()
        eat('-') -> value -= term()
        else -> return value
      }
    }
  }

  /** term := unary (('*' | '/' | '%') unary)* */
  private fun term(): Double {
    var value = unary()
    while (true) {
      when {
        eat('*') -> value *= unary()
        eat('/') -> {
          val divisor = unary()
          require(divisor != 0.0) { "division by zero" }
          value /= divisor
        }
        eat('%') -> {
          val divisor = unary()
          require(divisor != 0.0) { "division by zero" }
          value %= divisor
        }
        else -> return value
      }
    }
  }

  /** unary := ('-' | '+')? primary */
  private fun unary(): Double = when {
    eat('-') -> -unary()
    eat('+') -> unary()
    else -> primary()
  }

  /** primary := number | '(' expression ')' */
  private fun primary(): Double {
    skipSpace()
    if (eat('(')) {
      val value = expression()
      require(eat(')')) { "missing ')' at position $at" }
      return value
    }
    val start = at
    while (at < text.length && (text[at].isDigit() || text[at] == '.')) at++
    require(at > start) {
      if (at < text.length) "unexpected '${text[at]}' at position $at" else "expression ended early"
    }
    return text.substring(start, at).toDoubleOrNull()
      ?: throw IllegalArgumentException("'${text.substring(start, at)}' is not a number")
  }

  fun expectEnd() {
    skipSpace()
    require(at >= text.length) { "unexpected '${text[at]}' at position $at" }
  }

  private fun eat(symbol: Char): Boolean {
    skipSpace()
    if (at < text.length && text[at] == symbol) {
      at++
      return true
    }
    return false
  }

  private fun skipSpace() {
    while (at < text.length && text[at].isWhitespace()) at++
  }
}

/**
 * Formats a result for a model to read back.
 *
 * Whole numbers lose the `.0`, because a model handed "4.0" will often repeat it verbatim
 * into a sentence where "4" is what a person expects to read.
 */
fun formatArithmetic(value: Double): String = when {
  value.isNaN() || value.isInfinite() -> "undefined"
  value == value.toLong().toDouble() && kotlin.math.abs(value) < 1e15 -> value.toLong().toString()
  else -> value.toString()
}
