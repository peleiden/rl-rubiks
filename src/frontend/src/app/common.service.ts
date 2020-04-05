import { Injectable } from '@angular/core';
import { cube, cube20 } from './rubiks/rubiks';
import { HttpService } from './http.service';
import * as _ from "lodash";

@Injectable({
  providedIn: 'root'
})
export class CommonService {

  state: cube;
  state20: cube20;

  constructor(private httpService: HttpService) { }

  public async getSolved() {
    const { state, state20 } = await this.httpService.getSolved();
    this.state = state.map(val => _.flatten(val));
    this.state20 = state20;
    console.log(this.state);
  }

  public async act(action: number) {
    const { state, state20 } = await this.httpService.performAction({action, state20: this.state20});
    this.state = state.map(val => _.flatten(val));
    this.state20 = state20
  }
}
