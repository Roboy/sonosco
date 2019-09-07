<template>
  <md-card md-with-hover>
    <md-card-header>
      <div class="md-title">
        {{ name }}
      </div>
    </md-card-header>
    <md-card-content v-html="socketMessage"></md-card-content>
    <md-card-actions>
        <md-button @click="removeMyself()">Close</md-button>
    </md-card-actions>
  </md-card>
</template>

<script>
var jsdiff = require('diff')
import {bus} from '../main'

export default {
  name: 'Model',
  props: ['name', 'model_id'],
  data () {
    return {
      isConnected: false,
      socketMessage: 'This is a transcription.',
      recordButtonText: 'Record',
      refTranscription: 'This is a transcription',
      targetTranscription: 'This is a transcription'
    }
  },

  mounted () {
    bus.$on('updateTranscriptions', this.resetTranscription)
    bus.$on('updateRefTranscriptions', newRefTranscription => {
      this.refTranscription = newRefTranscription
      this.resetTranscription()
    })
  },

  sockets: {
    connect () {
      // Fired when the socket connects.
      this.isConnected = true
    },

    disconnect () {
      this.isConnected = false
    },

    // Fired when the server sends something on the "transcription" channel.
    transcription (data) {
      this.targetTranscription = data[this.model_id]
      if (Array.isArray(this.targetTranscription)) {
        this.targetTranscription = this.targetTranscription[0]
      }

      this.refTranscription = data['reference_id']
      if (Array.isArray(this.refTranscription)) {
        this.refTranscription = this.refTranscription[0]
      }

      this.resetTranscription()

      // if (Array.isArray(transcriptions)) {
      //   if (transcriptions.length === 1) {
      //     if ($parent.visualComparison) {
      //       this.socketMessage = this.getTranscriptionFormatted(transcriptions[0], data)
      //     } else {
      //       this.socketMessage = data
      //     }
      //     return
      //   }
      //
      //   var output = '<ol>'
      //
      //   for (var i = 0; i < transcriptions.length; i++) {
      //     output += '<li>' + transcriptions[i] + '</li>'
      //   }
      //
      //   output += '</ol>'
      //
      //   this.socketMessage = output
      // }
      // else {
      //   if ($parent.visualComparison) {
      //     this.socketMessage = this.getTranscriptionFormatted(transcriptions, data)
      //   } else {
      //     this.socketMessage = data
      //   }
      // }
    }
  },

  methods: {

    removeMyself () {
      this.$store.commit('removeModel', this.model_id)
    },

    resetTranscription () {
      if (this.$parent.visualComparison) {
        let diff = jsdiff.diffChars(this.refTranscription, this.targetTranscription)
        let final_transcription = '<span>'

        diff.forEach(function(part) {
          let color = part.added ? 'text-decoration:line-through' : part.removed ? 'color:red' : 'color:black'
          final_transcription += '<span style="' + color + '">' + part.value + '</span>'
        })

        final_transcription += '</span>'

        this.socketMessage = final_transcription
      } else {
        this.socketMessage = this.targetTranscription
      }
    }

  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  .Model {
    display: flex;
    flex-direction: column;
    border: solid;
    margin: 5px;
    min-height: 200px;
  }

  .Model h1 {
    text-align: center;
  }

  .Model p {
    padding: 1px 10px 1px 10px;
    flex-grow: 1;
    font-size: 15px;
    font-weight: bold;
  }
</style>
