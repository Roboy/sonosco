<template>
    <div class="ModelSelector">
        <md-field class="select">
            <label for="model">ModelSelector</label>
            <md-select id="model" v-model="chosenModel">
                <md-option v-for="(item, key, index) in models" v-bind:value="key" v-bind:key="key">
                  {{ item }}
                </md-option>
            </md-select>
        </md-field>
        <div class="controls">
            <md-button class="md-raised md-primary" @click="add(chosenModel)">Add</md-button>
            <md-button class="md-raised md-accent" @click="cancel">Cancel</md-button>
        </div>
    </div>
</template>

<script>
    import axios from 'axios'
    import {ServerAddress} from "../main";

    export default {
        name: 'ModelSelector',

        data() {
            return {
                models: null,
                chosenModel: null
            }
        },

        mounted() {
            var that = this
            axios
                .get(ServerAddress + "/get_models")
                .then(response => {
                    console.log("Models received from server: ", response.data)
                    that.models = response.data
                    that.chosenModel = response.data[0]
                })
                .catch(error => {
                    console.log("ERROR IN AXIOS", error)
                })
        },
        sockets: {
            connect() {
                // Fired when the socket connects.
                this.isConnected = true
                // this.$socket.emit('get_models')
            },

            disconnect() {
                this.isConnected = false
            },

            // Fired when the server sends something on the "transcription" channel.
            transcription(data) {
                this.socketMessage = data
            },

        },

        methods: {
            add(modelId) {
                this.$store.commit('addModel', {id: modelId, name: this.models[modelId]})
                this.$router.push('/')
            },

            cancel() {
                this.$router.push('/')
            }
        }
    }
</script>

<style scoped>
  .ModelSelector {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
  }

  .controls {
      margin: 20px 10px;
  }

  .select {
      margin: 0 6px;
      display: inline-flex;
      width: 50%;
  }

</style>
