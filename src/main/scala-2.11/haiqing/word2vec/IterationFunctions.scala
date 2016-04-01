package haiqing.word2vec

import com.github.fommil.netlib.BLAS.{getInstance => blas}

/**
 * Created by hwang on 04.03.16.
 */
class IterationFunctions (private val window: Int, private val vectorSize: Int, private val multiSense: Int,
                          private val senseCountAdd: Array[Array[Int]], private val senseCount: Array[Array[Int]],
                          private val expTable: Array[Float], private val senseTable: Array[Int],
                          private val syn0: Array[Array[Array[Float]]], private val syn1: Array[Array[Array[Float]]]) {
  private val MAX_EXP = 6
  private val U = 100
  private val ENCODE = 100

  private var sentence: Array[Int] = null
  private var sentenceNEG: Array[Array[Int]] = null

  def setSentence(sentence: Array[Int]): Unit = {
    this.sentence = sentence
  }

  def setSentenceNEG(sentenceNEG: Array[Array[Int]]): Unit = {
    this.sentenceNEG = sentenceNEG
  }

  def adjustSentence(): Boolean = {
    var flag = false
    for (pos <- 0 to sentence.size - 1) {
      val word = sentence(pos) / ENCODE
      if (senseTable(word) == multiSense) {
        var bestSense = -1 //there is no best sense
        var bestScore = 0.0
        for (sense <- 0 to senseTable(word) - 1) {
          val w = word * ENCODE + sense
          val score = getScore(w, pos)
          if (bestSense == -1 || score > bestScore) {
            bestScore = score
            bestSense = sense
          }
        }
        if (word * ENCODE + bestSense != sentence(pos)) {
          flag = true
        }
        sentence(pos) = word * ENCODE + bestSense
      }
    }
    flag
  }

  def sentenceLoss(): Double = {
    var loss = 0.0
    for (pos <- 0 to sentence.size - 1)
      loss += -getScore(sentence(pos),pos)
    loss
  }

  private def getScore(w: Int, pos: Int): Double = {
    var score = 0.0
    for (p <- pos-window+1 to pos+window-1)
      if (p != pos && p >=0 && p < sentence.size) {
        val u = sentence(p)
        val NEG = sentenceNEG(p)
        score += Math.log(activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(u/ENCODE)(u%ENCODE)))
        for (z <- NEG)
          score += Math.log(1 - activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(z/ENCODE)(z%ENCODE)))
      }
    score
  }

  def addSenseCount(): Unit = {
    for (w <- sentence) {
      val word = w / ENCODE
      val sense = w % ENCODE
      senseCountAdd(word)(sense) += 1
    }
  }

  def learnSentence(alpha: Float): Unit = {
    for (pos <- 0 to sentence.size - 1) {
      val w = sentence(pos)
      //val word = w / ENCODE
      //val sense = w % ENCODE
      //if (senseTable(word) == multiSense) {
        //if (multiSense == 1)
        //  learn(w, pos, alpha, alpha)
        //else
      learn(w, pos, alpha, alpha)
        /*
        for (otherSense <- 0 to senseTable(word) - 1)
          if (sense != otherSense && senseCount(word)(sense) > senseCount(word)(otherSense) * U) {
            val w = word * ENCODE + otherSense
            learn(w, pos, alpha / 2, alpha / 2)
          }
          */
      //}
    }
  }

  private def learn(w: Int, pos: Int, alphaW: Float, alphaU: Float): Unit = {
    for (p <- pos-window+1 to pos+window-1)
      if (p != pos && p >=0 && p < sentence.size) {
        val u = sentence(p)
        val NEG = sentenceNEG(p)
        val gradientW = new Array[Float](vectorSize)
        val g = (1 - activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(u/ENCODE)(u%ENCODE))).toFloat
        blas.saxpy(vectorSize, g, syn1(u/ENCODE)(u%ENCODE), 1, gradientW, 1)
        blas.saxpy(vectorSize, g*alphaU, syn0(w/ENCODE)(w%ENCODE), 1, syn1(u/ENCODE)(u%ENCODE), 1)
        for (z <- NEG) {
          val g = (-activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(z/ENCODE)(z%ENCODE))).toFloat
          blas.saxpy(vectorSize, g, syn1(z/ENCODE)(z%ENCODE), 1, gradientW, 1)
          blas.saxpy(vectorSize, g*alphaU, syn0(w/ENCODE)(w%ENCODE), 1, syn1(z/ENCODE)(z%ENCODE), 1)
        }
        blas.saxpy(vectorSize, alphaW, gradientW, 1, syn0(w/ENCODE)(w%ENCODE), 1)
    }
  }

  private def activeFunction(v0: Array[Float], v1: Array[Float]): Double = {
    val vectorSize = v0.length
    var f = blas.sdot(vectorSize, v0, 1, v1, 1)
    if (f > MAX_EXP)
      f = expTable(expTable.length - 1)
    else if (f < -MAX_EXP)
      f = expTable(0)
    else {
      val ind = ((f + MAX_EXP) * (expTable.size / MAX_EXP / 2.0)).toInt
      f = expTable(ind)
    }
    f
  }
}
