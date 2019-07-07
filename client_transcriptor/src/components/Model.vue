<template>
  <div class="Model">
    <h1>{{ name }}</h1>
    <p>{{ socketMessage }}</p>
  </div>
</template>

<script>
export default {
  name: 'Model',
  props: ['name'],
  data () {
    return {
      isConnected: false,
      socketMessage: 'This is a transcription.',
      recordButtonText: 'Record'
    }
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
      this.socketMessage = data[0]
    }
  },

  methods: {
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
