package haiqing.word2vec

import scala.collection.mutable.ArrayBuilder
import com.github.fommil.netlib.BLAS.{getInstance => blas}

/**
 * Created by hwang on 04.03.16.
 */
class IterationFunctions {
  var U = 0
  var ENCODE = 0
  var MAX_EXP = 0
  var alpha = 0.0f
  var negative = 0
  var vocabSize = 0
  var vectorSize = 0
  var window = 0
  var senseCountAdd: Array[Array[Int]] = null
  var senseCount: Array[Array[Int]] = null
  var expTable: Array[Float] = null
  var negTable: Array[Int] = null
  var senseTable: Array[Int] = null
  var syn0: Array[Array[Array[Float]]] = null
  var syn1: Array[Array[Array[Float]]] = null
  def addSenseCount(sentence: Array[Int]): Unit = {
    for (w <- sentence) {
      val word = w / ENCODE
      val sense = w % ENCODE
      if (word == 1115) {
        var a = 0
        a = 1
      }
      senseCountAdd(word)(sense) += 1
    }
  }
  def adjustSentence(sentence: Array[Int]): Boolean = {
    var flag = false
    for (pos <- 0 to sentence.size - 1) {
      val context = getNeighborsWithNEG(sentence, pos)
      val word = sentence(pos) / ENCODE
      var bestSense = -1  //there is no best sense
      var bestScore = 0.0
      for (sense <- 0 to senseTable(word)-1) {
        val w = word*ENCODE+sense
        val score = getScore(w,context)
        if (bestSense == -1 || score > bestScore) {
          bestScore = score
          bestSense = sense
        }
      }
      if (word*ENCODE+bestSense != sentence(pos)) {
        flag = true
      }
      sentence(pos) = word*ENCODE+bestSense
    }
    flag
  }
  def learnSentence(sentence: Array[Int]): Unit = {
    for (pos <- 0 to sentence.size - 1) {
      val context = getNeighborsWithNEG(sentence, pos)
      val w = sentence(pos)
      val word = w / ENCODE
      val sense = w % ENCODE
      learn(w, context, alpha, alpha)
      for (otherSense <- 0 to senseTable(word)-1)
        if (sense != otherSense && senseCount(word)(sense) > senseCount(word)(otherSense)*U) {
          val w = word*ENCODE+otherSense
          learn(w, context, alpha/2, 0)
        }
    }
  }
  def getNEG(word: Int): Array[Int] = {
    val negSamples = new Array[Int](negative)
    val tableSize = negTable.size
    for (i <- 0 to negative - 1) {
      negSamples(i) = word
      while (negSamples(i)/ENCODE == word/ENCODE) {
        negSamples(i) = negTable(Math.abs(util.Random.nextLong() % tableSize).toInt)
        if (negSamples(i) <= 0)
          negSamples(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
      }
      //add sense information (assign sense randomly)
      negSamples(i) = negSamples(i) * ENCODE + util.Random.nextInt(senseTable(negSamples(i)))
    }
    negSamples
  }
  def getNeighborsWithNEG(sentence: Array[Int], posW: Int): Array[(Int,Array[Int])] = {
    val neighbors = ArrayBuilder.make[(Int,Array[Int])]
    for (p <- posW - window + 1 to posW + window - 1)
      if (p >= 0 && p < sentence.size && p != posW)
        neighbors += sentence(p)->getNEG(sentence(p))
    neighbors.result()
  }
  def getScore(w: Int, context: Array[(Int,Array[Int])]): Double = {
    var score = 1.0
    for (token <- context) {
      val u = token._1
      val NEG = token._2
      score = score * activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(u/ENCODE)(u%ENCODE))
      for (z <- NEG)
        score = score * (1 - activeFunction(syn0(w/ENCODE)(w%ENCODE), syn1(z/ENCODE)(z%ENCODE)))
    }
    score
  }
  def learn(w: Int, context: Array[(Int,Array[Int])], alphaW: Float, alphaU: Float): Unit = {
    for (token <- context) {
      val u = token._1
      val NEG = token._2
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
  def activeFunction(v0: Array[Float], v1: Array[Float]): Double = {
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
